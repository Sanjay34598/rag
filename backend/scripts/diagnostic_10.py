import os
import sys
import json
import time
import numpy as np
from typing import List

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings, DATA_DIR
from app.services.retrieval.retrieval_service import get_retrieval_service

def pct(arr: List[float], p: float) -> float:
    if not arr:
        return 0.0
    return float(np.percentile(arr, p))

def main():
    eval_queries_path = os.path.join(DATA_DIR, "eval_queries.json")
    if not os.path.exists(eval_queries_path):
        print("eval_queries.json not found")
        return

    with open(eval_queries_path, "r", encoding="utf-8") as f:
        eval_queries = json.load(f)

    # 10 real queries
    sample_queries = [q["query"] for q in eval_queries[:10] if q.get("query")]
    
    print("Loading models and indexes...")
    ret_service = get_retrieval_service()
    
    # We will log if it's loading per query. It's using a singleton pattern.
    print(f"Is Embedding loaded once? {ret_service.embedding_service is not None}")
    
    # force load if not already
    ret_service.initialize(load_indexes=True)
    
    embed_ms = []
    faiss_ms = []
    bm25_ms = []
    fusion_ms = []
    rerank_ms = []
    total_ms = []

    print("Running 10 queries...")
    for q in sample_queries:
        # Retrieve logs inside will show if it's loading repeatedly, but it's not because initialize() is called once
        res = ret_service.retrieve(q, candidate_k=20, top_k=5, reranker_enabled=True)
        
        lb = res["latency_breakdown"]
        embed_ms.append(lb["embedding_ms"])
        faiss_ms.append(lb["faiss_ms"])
        bm25_ms.append(lb["bm25_ms"])
        fusion_ms.append(lb["fusion_ms"])
        rerank_ms.append(lb["reranking_ms"])
        total_ms.append(res["latency_ms"])
        
    print("\n--- RESULTS (10 Queries) ---")
    print(f"Embedding: {np.mean(embed_ms):.2f} ms (Avg) | {pct(embed_ms, 50):.2f} ms (P50) | {pct(embed_ms, 100):.2f} ms (P100)")
    print(f"FAISS: {np.mean(faiss_ms):.2f} ms (Avg) | {pct(faiss_ms, 50):.2f} ms (P50) | {pct(faiss_ms, 100):.2f} ms (P100)")
    print(f"BM25: {np.mean(bm25_ms):.2f} ms (Avg) | {pct(bm25_ms, 50):.2f} ms (P50) | {pct(bm25_ms, 100):.2f} ms (P100)")
    print(f"Fusion: {np.mean(fusion_ms):.2f} ms (Avg) | {pct(fusion_ms, 50):.2f} ms (P50) | {pct(fusion_ms, 100):.2f} ms (P100)")
    print(f"Reranking: {np.mean(rerank_ms):.2f} ms (Avg) | {pct(rerank_ms, 50):.2f} ms (P50) | {pct(rerank_ms, 100):.2f} ms (P100)")
    print(f"Total: {np.mean(total_ms):.2f} ms (Avg) | {pct(total_ms, 50):.2f} ms (P50) | {pct(total_ms, 100):.2f} ms (P100)")
    
    print("\n--- BOTTLENECK ANALYSIS ---")
    avgs = {
        "Embedding": np.mean(embed_ms),
        "FAISS": np.mean(faiss_ms),
        "BM25": np.mean(bm25_ms),
        "Fusion": np.mean(fusion_ms),
        "Reranking": np.mean(rerank_ms)
    }
    biggest = max(avgs, key=avgs.get)
    print(f"The single biggest bottleneck is: {biggest} taking {avgs[biggest]:.2f} ms on average.")

if __name__ == '__main__':
    main()
