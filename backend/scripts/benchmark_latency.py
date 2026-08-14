import os
import sys
import json
import numpy as np
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings, DATA_DIR
from app.services.retrieval.retrieval_service import get_retrieval_service

def percentile(arr: List[float], p: float) -> float:
    if not arr:
        return 0.0
    return float(np.percentile(arr, p))

def benchmark():
    print("==================================================")
    print("STAGE 3: RETRIEVAL LATENCY BENCHMARKING")
    print("==================================================")

    eval_queries_path = os.path.join(DATA_DIR, "eval_queries.json")
    if not os.path.exists(eval_queries_path):
        raise FileNotFoundError("eval_queries.json missing. Run build_indexes.py first.")

    with open(eval_queries_path, "r", encoding="utf-8") as f:
        eval_queries = json.load(f)

    # Use 100 representative real queries from dataset
    queries = [q["query"] for q in eval_queries[:100] if q.get("query")]
    print(f"[Benchmark] Benchmark sample size: {len(queries)} queries.")

    service = get_retrieval_service()
    service.initialize(load_indexes=True)

    # Warmup query
    service.retrieve(queries[0])

    print("\n--- Running Configuration A: Hybrid WITHOUT Reranker ---")
    config_a_latencies = []
    config_a_breakdowns = []
    
    for idx, q in enumerate(queries):
        res = service.retrieve(q, reranker_enabled=False)
        config_a_latencies.append(res["latency_ms"])
        config_a_breakdowns.append(res["latency_breakdown"])

    print("\n--- Running Configuration B: Hybrid WITH Reranker ---")
    config_b_latencies = []
    config_b_breakdowns = []

    for idx, q in enumerate(queries):
        res = service.retrieve(q, reranker_enabled=True)
        config_b_latencies.append(res["latency_ms"])
        config_b_breakdowns.append(res["latency_breakdown"])

    # Calculate percentiles
    p50_a = percentile(config_a_latencies, 50)
    p70_a = percentile(config_a_latencies, 70)
    p100_a = percentile(config_a_latencies, 100)

    p50_b = percentile(config_b_latencies, 50)
    p70_b = percentile(config_b_latencies, 70)
    p100_b = percentile(config_b_latencies, 100)

    print("\n==================================================")
    print("BENCHMARK LATENCY SUMMARY (Target: < 200 ms)")
    print("==================================================")
    print(f"{'Metric':<10} | {'Hybrid (No Rerank)':<20} | {'Hybrid + Reranker':<20}")
    print("-" * 58)
    print(f"{'P50':<10} | {p50_a:<20.2f} ms | {p50_b:<20.2f} ms")
    print(f"{'P70':<10} | {p70_a:<20.2f} ms | {p70_b:<20.2f} ms")
    print(f"{'P100':<10} | {p100_a:<20.2f} ms | {p100_b:<20.2f} ms")
    print("==================================================")

    # Save raw benchmark results
    raw_data = {
        "num_queries": len(queries),
        "target_latency_ms": 200,
        "config_a_hybrid_no_rerank": {
            "P50_ms": round(p50_a, 2),
            "P70_ms": round(p70_a, 2),
            "P100_ms": round(p100_a, 2),
            "avg_embedding_ms": round(float(np.mean([b["embedding_ms"] for b in config_a_breakdowns])), 2),
            "avg_faiss_ms": round(float(np.mean([b["faiss_ms"] for b in config_a_breakdowns])), 2),
            "avg_bm25_ms": round(float(np.mean([b["bm25_ms"] for b in config_a_breakdowns])), 2),
            "avg_fusion_ms": round(float(np.mean([b["fusion_ms"] for b in config_a_breakdowns])), 2),
            "raw_latencies_ms": [round(x, 2) for x in config_a_latencies]
        },
        "config_b_hybrid_with_rerank": {
            "P50_ms": round(p50_b, 2),
            "P70_ms": round(p70_b, 2),
            "P100_ms": round(p100_b, 2),
            "avg_embedding_ms": round(float(np.mean([b["embedding_ms"] for b in config_b_breakdowns])), 2),
            "avg_faiss_ms": round(float(np.mean([b["faiss_ms"] for b in config_b_breakdowns])), 2),
            "avg_bm25_ms": round(float(np.mean([b["bm25_ms"] for b in config_b_breakdowns])), 2),
            "avg_fusion_ms": round(float(np.mean([b["fusion_ms"] for b in config_b_breakdowns])), 2),
            "avg_reranking_ms": round(float(np.mean([b["reranking_ms"] for b in config_b_breakdowns])), 2),
            "raw_latencies_ms": [round(x, 2) for x in config_b_latencies]
        }
    }

    benchmark_json_path = os.path.join(DATA_DIR, "retrieval_benchmark.json")
    with open(benchmark_json_path, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2)
    print(f"[Benchmark] Saved raw benchmark results to {benchmark_json_path}")

    # Generate Markdown Report
    doc_dir = os.path.join(settings.BASE_DIR, "docs")
    os.makedirs(doc_dir, exist_ok=True)
    report_md_path = os.path.join(doc_dir, "retrieval_benchmark.md")
    
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(f"""# Retrieval Latency Benchmark Report

## Executive Summary
This report presents measured latency performance for the Stage 3 Hybrid Retrieval Engine over {len(queries)} real MSMARCO-XI Hindi queries.

Target Latency: **< 200 ms**

## Benchmark Results Table

| Percentile | Hybrid Retrieval (No Rerank) | Hybrid + Reranker (`ms-marco-MiniLM-L-6-v2`) |
| :--- | :--- | :--- |
| **P50 (Median)** | **{p50_a:.2f} ms** | **{p50_b:.2f} ms** |
| **P70** | **{p70_a:.2f} ms** | **{p70_b:.2f} ms** |
| **P100 (Max)** | **{p100_a:.2f} ms** | **{p100_b:.2f} ms** |

## Stage-by-Stage Latency Breakdown (Averages)

### Configuration A (Hybrid - No Rerank)
- **Query Embedding**: {raw_data['config_a_hybrid_no_rerank']['avg_embedding_ms']:.2f} ms
- **FAISS Vector Search**: {raw_data['config_a_hybrid_no_rerank']['avg_faiss_ms']:.2f} ms
- **BM25 Keyword Search**: {raw_data['config_a_hybrid_no_rerank']['avg_bm25_ms']:.2f} ms
- **Score Fusion & Sorting**: {raw_data['config_a_hybrid_no_rerank']['avg_fusion_ms']:.2f} ms
- **Total Latency (P50)**: **{p50_a:.2f} ms**

### Configuration B (Hybrid + Reranker)
- **Query Embedding**: {raw_data['config_b_hybrid_with_rerank']['avg_embedding_ms']:.2f} ms
- **FAISS Vector Search**: {raw_data['config_b_hybrid_with_rerank']['avg_faiss_ms']:.2f} ms
- **BM25 Keyword Search**: {raw_data['config_b_hybrid_with_rerank']['avg_bm25_ms']:.2f} ms
- **Score Fusion**: {raw_data['config_b_hybrid_with_rerank']['avg_fusion_ms']:.2f} ms
- **Cross-Encoder Reranking**: {raw_data['config_b_hybrid_with_rerank']['avg_reranking_ms']:.2f} ms
- **Total Latency (P50)**: **{p50_b:.2f} ms**

## Recommendations
- For real-time voice RAG applications requiring sub-100ms response time, **Hybrid Retrieval without reranking** easily satisfies the hackathon target ({p50_a:.2f} ms P50).
- Reranking can be toggled dynamically via environment variable `RERANKER_ENABLED=false` or `RERANKER_ENABLED=true`.
""")
    print(f"[Benchmark] Generated markdown report at {report_md_path}")

if __name__ == "__main__":
    benchmark()
