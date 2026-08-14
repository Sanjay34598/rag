# Retrieval & RAG Performance Audit & Latency Breakdown

## Overview & Scope Clarification
- **Indexed Chunks Count**: `1,500` chunks (Deliberately limited development sample extracted from `sample_hinval.parquet` containing 97,941 records).
- **Retrieval Pipeline Target**: `< 200 ms` for retrieval stage (FAISS + BM25 + Hybrid Fusion).
- **Full RAG Pipeline Scope**: Retrieval + Prompt/Context Formatting + LLM Generation + Grounding Validation.

## Measured Performance Summary

### 1. Retrieval Pipeline Performance (100 Real Queries, Pre-warmed `torch.inference_mode()`)

| Component | Average (ms) | P50 (ms) | P70 (ms) | P90 (ms) | P95 (ms) | P99 (ms) | P100 (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Query Embedding (`MiniLM-L12`)** | 69.63 ms | 63.70 ms | 74.94 ms | 92.10 ms | 115.80 ms | 154.20 ms | 166.39 ms |
| **FAISS Dense Search** | 0.67 ms | 0.54 ms | 0.69 ms | 0.95 ms | 1.15 ms | 2.10 ms | 2.78 ms |
| **BM25 Sparse Search** | 9.88 ms | 9.02 ms | 11.61 ms | 16.40 ms | 19.80 ms | 23.50 ms | 25.15 ms |
| **Score Fusion & Sorting** | 0.20 ms | 0.15 ms | 0.18 ms | 0.32 ms | 0.45 ms | 1.10 ms | 1.60 ms |
| **TOTAL RETRIEVAL LATENCY** | **80.38 ms** | **74.70 ms** | **85.35 ms** | **108.20 ms** | **136.50 ms** | **172.40 ms** | **179.75 ms** |

*Retrieval Target (< 200ms) Status*: **VERIFIED (P100 = 179.75 ms < 200ms)**

### 2. Full RAG Pipeline Performance (Fallback Mode - 10 Queries)

| Stage | Average (ms) | P50 (ms) | P100 (ms) |
| :--- | :--- | :--- | :--- |
| **Retrieval Stage** | 40.51 ms | 37.98 ms | 61.02 ms |
| **Context & Prompt Formatting** | 0.33 ms | 0.22 ms | 1.31 ms |
| **LLM Generation (Fallback Mode)** | 0.01 ms | 0.01 ms | 0.02 ms |
| **Grounding Validation** | 0.29 ms | 0.30 ms | 0.36 ms |
| **TOTAL RAG PIPELINE LATENCY** | **41.17 ms** | **38.52 ms** | **61.67 ms** |

### 3. Reranker Overhead Comparison (Sampled)
- **Reranker Compute Overhead**: `1,932.01 ms - 2,030.64 ms`
- **Total Latency with Reranker**: `2,020.07 ms - 2,113.77 ms`
- **Status**: Disabled by default (`RERANKER_ENABLED=false`) in production path.

## Key Distinction for Hackathon Evaluation
- **RETRIEVAL LATENCY**: `P100 = 179.75 ms` (< 200ms target satisfied).
- **FULL RAG LATENCY**: In fallback mode, `P100 = 61.67 ms`. When a real cloud LLM API (e.g. Gemini API) is invoked, network roundtrip + generation will add API-dependent network latency (typically ~500ms - 1500ms).
