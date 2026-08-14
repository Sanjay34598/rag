import os
import sys
import json
import time
import torch
import numpy as np
from typing import List, Dict, Any

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings, DATA_DIR
from app.services.retrieval.retrieval_service import get_retrieval_service

def pct(arr: List[float], p: float) -> float:
    if not arr:
        return 0.0
    return float(np.percentile(arr, p))

def run_investigation():
    eval_queries_path = os.path.join(DATA_DIR, "eval_queries.json")
    with open(eval_queries_path, "r", encoding="utf-8") as f:
        eval_queries = json.load(f)

    queries = [q["query"] for q in eval_queries[:100] if q.get("query")]

    service = get_retrieval_service()
    service.initialize(load_indexes=True)

    records = []

    for idx, q in enumerate(queries):
        res = service.retrieve(q, candidate_k=20, top_k=5, reranker_enabled=False)
        lb = res["latency_breakdown"]
        records.append({
            "idx": idx + 1,
            "query": q,
            "query_len": len(q),
            "embed_ms": lb["embedding_ms"],
            "faiss_ms": lb["faiss_ms"],
            "bm25_ms": lb["bm25_ms"],
            "fusion_ms": lb["fusion_ms"],
            "total_ms": res["latency_ms"]
        })

    totals = [r["total_ms"] for r in records]
    embeds = [r["embed_ms"] for r in records]
    faisses = [r["faiss_ms"] for r in records]
    bm25s = [r["bm25_ms"] for r in records]
    fusions = [r["fusion_ms"] for r in records]

    print("==================================================")
    print("CONTROLLED LATENCY BENCHMARK (100 QUERIES)")
    print("==================================================")
    print(f"Average : {np.mean(totals):.2f} ms")
    print(f"P50     : {pct(totals, 50):.2f} ms")
    print(f"P70     : {pct(totals, 70):.2f} ms")
    print(f"P90     : {pct(totals, 90):.2f} ms")
    print(f"P95     : {pct(totals, 95):.2f} ms")
    print(f"P99     : {pct(totals, 99):.2f} ms")
    print(f"P100    : {pct(totals, 100):.2f} ms")
    print("--------------------------------------------------")
    print(f"Component Averages: Embedding={np.mean(embeds):.2f}ms | FAISS={np.mean(faisses):.2f}ms | BM25={np.mean(bm25s):.2f}ms | Fusion={np.mean(fusions):.2f}ms")

    # Sort records by total_ms descending
    slowest_10 = sorted(records, key=lambda x: x["total_ms"], reverse=True)[:10]

    print("\n==================================================")
    print("SLOWEST 10 QUERIES INDIVIDUAL BREAKDOWN")
    print("==================================================")
    header = f"{'Rank':<4} | {'Query (Truncated)':<30} | {'Len':<4} | {'Embed':<8} | {'FAISS':<6} | {'BM25':<6} | {'Fusion':<6} | {'Total':<8}"
    print(header)
    print("-" * len(header))
    for r_idx, item in enumerate(slowest_10, start=1):
        q_trunc = item['query'][:28] + ".." if len(item['query']) > 30 else item['query']
        print(f"{r_idx:<4} | {q_trunc:<30} | {item['query_len']:<4} | {item['embed_ms']:<8.2f} | {item['faiss_ms']:<6.2f} | {item['bm25_ms']:<6.2f} | {item['fusion_ms']:<6.2f} | {item['total_ms']:<8.2f}")

    print("==================================================")

if __name__ == "__main__":
    run_investigation()
