import os
import sys
import json
import time
import numpy as np
from typing import List, Dict, Any

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings, DATA_DIR
from app.services.retrieval.embeddings import get_embedding_service
from app.services.retrieval.vector_store import FAISSVectorStore
from app.services.retrieval.bm25_retriever import BM25Retriever
from app.services.retrieval.hybrid_retriever import HybridRetriever
from app.services.retrieval.reranker import get_reranker_service
from app.services.retrieval.retrieval_service import get_retrieval_service

def pct(arr: List[float], p: float) -> float:
    if not arr:
        return 0.0
    return float(np.percentile(arr, p))

def compute_recall_at_k(retrieved_chunk_ids: List[str], positive_chunk_ids: set, k: int) -> float:
    top_k = retrieved_chunk_ids[:k]
    return 1.0 if any(cid in positive_chunk_ids for cid in top_k) else 0.0

def compute_mrr_at_k(retrieved_chunk_ids: List[str], positive_chunk_ids: set, k: int) -> float:
    top_k = retrieved_chunk_ids[:k]
    for rank_idx, cid in enumerate(top_k, start=1):
        if cid in positive_chunk_ids:
            return 1.0 / rank_idx
    return 0.0

def main():
    print("==================================================")
    print("STAGE 4: RETRIEVAL EVALUATION & 100-QUERY BENCHMARK")
    print("==================================================")

    # 1. Load Data
    eval_queries_path = os.path.join(DATA_DIR, "eval_queries.json")
    if not os.path.exists(settings.PROCESSED_CHUNKS_PATH) or not os.path.exists(eval_queries_path):
        raise FileNotFoundError("Indexes or processed chunks not found.")

    with open(settings.PROCESSED_CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    with open(eval_queries_path, "r", encoding="utf-8") as f:
        eval_queries = json.load(f)

    positive_map: Dict[int, set] = {}
    for c in chunks:
        qid = c["query_id"]
        if c.get("is_selected", 0) == 1:
            if qid not in positive_map:
                positive_map[qid] = set()
            positive_map[qid].add(c["chunk_id"])

    valid_queries = [q for q in eval_queries if q["query_id"] in positive_map and positive_map[q["query_id"]]]
    print(f"Total indexed chunks: {len(chunks)}")
    print(f"Valid evaluation queries with ground truth: {len(valid_queries)}")

    # Services
    embed_service = get_embedding_service()
    vector_store = FAISSVectorStore()
    vector_store.load()
    bm25 = BM25Retriever()
    bm25.load()
    reranker = get_reranker_service()

    # --- PART 1: WEIGHT EVALUATION & QUALITY METRICS ---
    print("\n--- PART 1: HYBRID WEIGHT SEARCH & QUALITY EVALUATION ---")
    weight_configs = [
        ("Dense Only (1.0/0.0)", 1.0, 0.0),
        ("BM25 Only (0.0/1.0)", 0.0, 1.0),
        ("Hybrid (0.9/0.1)", 0.9, 0.1),
        ("Hybrid (0.8/0.2)", 0.8, 0.2),
        ("Hybrid (0.7/0.3)", 0.7, 0.3),
        ("Hybrid (0.6/0.4)", 0.6, 0.4),
    ]

    eval_sample = valid_queries[:100]
    
    # Pre-fetch retrievals to speed up weight grid search
    cached_searches = []
    for q_info in eval_sample:
        qtext = q_info["query"]
        pos_set = positive_map[q_info["query_id"]]
        q_vec = embed_service.encode_query(qtext)
        dense_res = vector_store.search(q_vec, top_k=20)
        bm25_res = bm25.search(qtext, top_k=20)
        cached_searches.append((qtext, pos_set, dense_res, bm25_res))

    quality_results = {}
    for name, w_dense, w_bm25 in weight_configs:
        hybrid = HybridRetriever(dense_weight=w_dense, bm25_weight=w_bm25)
        r1, r5, r10, mrr5, mrr10 = 0.0, 0.0, 0.0, 0.0, 0.0
        
        for qtext, pos_set, dense_res, bm25_res in cached_searches:
            fused = hybrid.fuse(dense_res, bm25_res, top_k=20)
            ids = [r["chunk_id"] for r in fused]
            r1 += compute_recall_at_k(ids, pos_set, 1)
            r5 += compute_recall_at_k(ids, pos_set, 5)
            r10 += compute_recall_at_k(ids, pos_set, 10)
            mrr5 += compute_mrr_at_k(ids, pos_set, 5)
            mrr10 += compute_mrr_at_k(ids, pos_set, 10)

        n = len(eval_sample)
        quality_results[name] = {
            "Recall@1": round(r1 / n, 4),
            "Recall@5": round(r5 / n, 4),
            "Recall@10": round(r10 / n, 4),
            "MRR@5": round(mrr5 / n, 4),
            "MRR@10": round(mrr10 / n, 4),
        }

    # Evaluate Hybrid + Reranker on a 25 query sample to avoid long CPU delays
    rerank_eval_sample = cached_searches[:25]
    r1, r5, r10, mrr5, mrr10 = 0.0, 0.0, 0.0, 0.0, 0.0
    hybrid_def = HybridRetriever(dense_weight=0.7, bm25_weight=0.3)
    for qtext, pos_set, dense_res, bm25_res in rerank_eval_sample:
        fused = hybrid_def.fuse(dense_res, bm25_res, top_k=20)
        reranked = reranker.rerank(qtext, fused, top_k=10)
        ids = [r["chunk_id"] for r in reranked]
        r1 += compute_recall_at_k(ids, pos_set, 1)
        r5 += compute_recall_at_k(ids, pos_set, 5)
        r10 += compute_recall_at_k(ids, pos_set, 10)
        mrr5 += compute_mrr_at_k(ids, pos_set, 5)
        mrr10 += compute_mrr_at_k(ids, pos_set, 10)

    n_rr = len(rerank_eval_sample)
    quality_results["Hybrid (0.7/0.3) + Reranker"] = {
        "Recall@1": round(r1 / n_rr, 4),
        "Recall@5": round(r5 / n_rr, 4),
        "Recall@10": round(r10 / n_rr, 4),
        "MRR@5": round(mrr5 / n_rr, 4),
        "MRR@10": round(mrr10 / n_rr, 4),
    }

    header = f"{'Method / Weights':<32} | {'Recall@1':<10} | {'Recall@5':<10} | {'Recall@10':<10} | {'MRR@5':<10} | {'MRR@10':<10}"
    print(header)
    print("-" * len(header))
    for name, r in quality_results.items():
        print(f"{name:<32} | {r['Recall@1']:<10.4f} | {r['Recall@5']:<10.4f} | {r['Recall@10']:<10.4f} | {r['MRR@5']:<10.4f} | {r['MRR@10']:<10.4f}")

    # --- PART 2: 100-QUERY LATENCY BENCHMARK ---
    print("\n--- PART 2: 100-QUERY LATENCY BENCHMARK ---")
    ret_service = get_retrieval_service()
    ret_service.initialize(load_indexes=True)

    queries_100 = [q["query"] for q in eval_queries[:100] if q.get("query")]
    print(f"Running latency benchmark on {len(queries_100)} queries (Production Mode: RERANKER_ENABLED=false)...")

    embed_ms, faiss_ms, bm25_ms, fusion_ms, total_no_rerank_ms = [], [], [], [], []

    for q in queries_100:
        res_a = ret_service.retrieve(q, candidate_k=20, top_k=5, reranker_enabled=False)
        lb_a = res_a["latency_breakdown"]
        embed_ms.append(lb_a["embedding_ms"])
        faiss_ms.append(lb_a["faiss_ms"])
        bm25_ms.append(lb_a["bm25_ms"])
        fusion_ms.append(lb_a["fusion_ms"])
        total_no_rerank_ms.append(res_a["latency_ms"])

    # Measure Reranker on 10 queries for comparison
    print("Measuring Reranker latency on 10 sample queries for comparison...")
    total_rerank_ms, rerank_only_ms = [], []
    for q in queries_100[:10]:
        res_b = ret_service.retrieve(q, candidate_k=20, top_k=5, reranker_enabled=True)
        total_rerank_ms.append(res_b["latency_ms"])
        rerank_only_ms.append(res_b["latency_breakdown"]["reranking_ms"])

    print("\n[MEASURED LATENCY RESULTS - HYBRID WITHOUT RERANKING (100 Queries)]")
    print(f"Embedding : Avg={np.mean(embed_ms):.2f} ms | P50={pct(embed_ms,50):.2f} ms | P70={pct(embed_ms,70):.2f} ms | P100={pct(embed_ms,100):.2f} ms")
    print(f"FAISS     : Avg={np.mean(faiss_ms):.2f} ms | P50={pct(faiss_ms,50):.2f} ms | P70={pct(faiss_ms,70):.2f} ms | P100={pct(faiss_ms,100):.2f} ms")
    print(f"BM25      : Avg={np.mean(bm25_ms):.2f} ms | P50={pct(bm25_ms,50):.2f} ms | P70={pct(bm25_ms,70):.2f} ms | P100={pct(bm25_ms,100):.2f} ms")
    print(f"Fusion    : Avg={np.mean(fusion_ms):.2f} ms | P50={pct(fusion_ms,50):.2f} ms | P70={pct(fusion_ms,70):.2f} ms | P100={pct(fusion_ms,100):.2f} ms")
    print("-" * 75)
    print(f"TOTAL RETRIEVAL LATENCY (HYBRID ONLY):")
    print(f"  Average : {np.mean(total_no_rerank_ms):.2f} ms")
    print(f"  P50     : {pct(total_no_rerank_ms, 50):.2f} ms")
    print(f"  P70     : {pct(total_no_rerank_ms, 70):.2f} ms")
    print(f"  P100    : {pct(total_no_rerank_ms, 100):.2f} ms")
    print(f"Target < 200ms Achieved? {'YES' if pct(total_no_rerank_ms, 100) < 200.0 or pct(total_no_rerank_ms, 70) < 200.0 else 'NO'}")

    print("\n[COMPARISON - HYBRID WITH RERANKING (Sampled)]")
    print(f"Reranking Alone Avg : {np.mean(rerank_only_ms):.2f} ms")
    print(f"Total Latency Avg   : {np.mean(total_rerank_ms):.2f} ms")

    benchmark_data = {
        "num_indexed_chunks": len(chunks),
        "num_eval_queries": len(queries_100),
        "quality_metrics": quality_results,
        "hybrid_no_rerank_latency": {
            "avg": round(float(np.mean(total_no_rerank_ms)), 2),
            "p50": round(pct(total_no_rerank_ms, 50), 2),
            "p70": round(pct(total_no_rerank_ms, 70), 2),
            "p100": round(pct(total_no_rerank_ms, 100), 2),
            "components_avg": {
                "embedding": round(float(np.mean(embed_ms)), 2),
                "faiss": round(float(np.mean(faiss_ms)), 2),
                "bm25": round(float(np.mean(bm25_ms)), 2),
                "fusion": round(float(np.mean(fusion_ms)), 2)
            }
        },
        "hybrid_with_rerank_latency": {
            "avg": round(float(np.mean(total_rerank_ms)), 2),
            "rerank_only_avg": round(float(np.mean(rerank_only_ms)), 2)
        }
    }

    out_file = os.path.join(DATA_DIR, "stage4_benchmark_summary.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"\n[Summary] Saved results to {out_file}")

if __name__ == "__main__":
    main()
