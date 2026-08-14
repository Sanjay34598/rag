import os
import sys
import json
import time
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings, DATA_DIR
from app.services.retrieval.embeddings import get_embedding_service
from app.services.retrieval.vector_store import FAISSVectorStore
from app.services.retrieval.bm25_retriever import BM25Retriever
from app.services.retrieval.hybrid_retriever import HybridRetriever
from app.services.retrieval.reranker import get_reranker_service

def compute_recall_at_k(retrieved_chunk_ids: List[str], positive_chunk_ids: set, k: int) -> float:
    top_k = retrieved_chunk_ids[:k]
    return 1.0 if any(cid in positive_chunk_ids for cid in top_k) else 0.0

def compute_mrr_at_k(retrieved_chunk_ids: List[str], positive_chunk_ids: set, k: int) -> float:
    top_k = retrieved_chunk_ids[:k]
    for rank_idx, cid in enumerate(top_k, start=1):
        if cid in positive_chunk_ids:
            return 1.0 / rank_idx
    return 0.0

def evaluate():
    print("==================================================")
    print("STAGE 3: RETRIEVAL QUALITY EVALUATION (RECALL & MRR)")
    print("==================================================")

    # 1. Load Processed Chunks & Eval Queries
    eval_queries_path = os.path.join(DATA_DIR, "eval_queries.json")
    if not os.path.exists(settings.PROCESSED_CHUNKS_PATH) or not os.path.exists(eval_queries_path):
        raise FileNotFoundError("Indexes or processed chunks not found. Run build_indexes.py first.")

    with open(settings.PROCESSED_CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    with open(eval_queries_path, "r", encoding="utf-8") as f:
        eval_queries = json.load(f)

    # Group ground truth positive chunks by query_id
    positive_map: Dict[int, set] = {}
    for c in chunks:
        qid = c["query_id"]
        if c.get("is_selected", 0) == 1:
            if qid not in positive_map:
                positive_map[qid] = set()
            positive_map[qid].add(c["chunk_id"])

    # Filter eval queries that have positive ground truth passages
    valid_queries = [q for q in eval_queries if q["query_id"] in positive_map and positive_map[q["query_id"]]]
    print(f"[Evaluation] Total queries: {len(eval_queries)}, Valid eval queries with positive ground truth: {len(valid_queries)}")

    # Limit evaluation sample size to 200 for fast reproducible evaluation
    sample_queries = valid_queries[:200]
    print(f"[Evaluation] Evaluating sample of {len(sample_queries)} representative queries...")

    # Load Services & Indexes
    embed_service = get_embedding_service()
    vector_store = FAISSVectorStore()
    vector_store.load()
    bm25 = BM25Retriever()
    bm25.load()
    hybrid = HybridRetriever()
    reranker = get_reranker_service()

    # Track metrics
    methods = ["dense", "bm25", "hybrid", "hybrid_rerank"]
    metrics = {m: {"r1": 0.0, "r5": 0.0, "r10": 0.0, "mrr5": 0.0, "mrr10": 0.0} for m in methods}

    start_t = time.time()
    for idx, q_info in enumerate(sample_queries):
        qid = q_info["query_id"]
        qtext = q_info["query"]
        pos_set = positive_map[qid]

        # Query embedding
        q_vec = embed_service.encode_query(qtext)
        dense_res = vector_store.search(q_vec, top_k=20)
        bm25_res = bm25.search(qtext, top_k=20)
        hybrid_res = hybrid.fuse(dense_res, bm25_res, top_k=20)
        rerank_res = reranker.rerank(qtext, hybrid_res, top_k=10)

        retrieved_ids = {
            "dense": [r["chunk_id"] for r in dense_res],
            "bm25": [r["chunk_id"] for r in bm25_res],
            "hybrid": [r["chunk_id"] for r in hybrid_res],
            "hybrid_rerank": [r["chunk_id"] for r in rerank_res]
        }

        for m in methods:
            ids = retrieved_ids[m]
            metrics[m]["r1"] += compute_recall_at_k(ids, pos_set, 1)
            metrics[m]["r5"] += compute_recall_at_k(ids, pos_set, 5)
            metrics[m]["r10"] += compute_recall_at_k(ids, pos_set, 10)
            metrics[m]["mrr5"] += compute_mrr_at_k(ids, pos_set, 5)
            metrics[m]["mrr10"] += compute_mrr_at_k(ids, pos_set, 10)

        if (idx + 1) % 50 == 0:
            print(f"[Evaluation] Processed {idx + 1}/{len(sample_queries)} queries...")

    eval_time = time.time() - start_t
    n = len(sample_queries)

    results_report = {}
    for m in methods:
        results_report[m] = {
            "Recall@1": round(metrics[m]["r1"] / n, 4),
            "Recall@5": round(metrics[m]["r5"] / n, 4),
            "Recall@10": round(metrics[m]["r10"] / n, 4),
            "MRR@5": round(metrics[m]["mrr5"] / n, 4),
            "MRR@10": round(metrics[m]["mrr10"] / n, 4)
        }

    print("\n==================================================")
    print("RETRIEVAL EVALUATION RESULTS REPORT")
    print("==================================================")
    print(f"Evaluated Queries Count: {n}")
    print(f"Evaluation Time:        {eval_time:.2f} seconds\n")
    
    header = f"{'Method':<18} | {'Recall@1':<10} | {'Recall@5':<10} | {'Recall@10':<10} | {'MRR@5':<10} | {'MRR@10':<10}"
    print(header)
    print("-" * len(header))
    
    for m in methods:
        r = results_report[m]
        print(f"{m:<18} | {r['Recall@1']:<10.4f} | {r['Recall@5']:<10.4f} | {r['Recall@10']:<10.4f} | {r['MRR@5']:<10.4f} | {r['MRR@10']:<10.4f}")
        
    print("==================================================")

    # Save report to JSON
    out_path = os.path.join(DATA_DIR, "retrieval_eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "evaluated_queries_count": n,
            "evaluation_time_seconds": round(eval_time, 2),
            "results": results_report
        }, f, indent=2)
    print(f"[Evaluation] Saved report to {out_path}")

if __name__ == "__main__":
    evaluate()
