# High-Performance Hybrid Retrieval Architecture

## Overview
The retrieval pipeline delivers relevant knowledge chunks from the indexed MSMARCO-XI dataset within strict latency budgets (< 200ms target for retrieval).

## Architecture Details

### Production Pipeline (`RERANKER_ENABLED=false`)
The default production path bypasses cross-encoder reranking to ensure real-time response speeds.

```
User Query
    ↓
Embedding (`paraphrase-multilingual-MiniLM-L12-v2`)
    ↓
Dense FAISS Search + Sparse BM25 Search
    ↓
Min-Max Score Normalization & Hybrid Fusion (Dense 0.7 / BM25 0.3)
    ↓
Top-K Results Output
```

### Reranking Module (`RERANKER_ENABLED=true`)
- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Purpose**: Provides a second-stage cross-attention reranking over hybrid candidates.
- **Latency Impact**: On the local CPU environment, evaluating 20 candidates per query adds ~1.9s to 2.7s of latency.
- **Production Status**: **Disabled by default** (`RERANKER_ENABLED=false`) for real-time applications. Preserved in codebase for offline evaluation, research, and future GPU/ONNX deployment.

## Production Configuration
- `EMBEDDING_MODEL`: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- `DENSE_WEIGHT`: `0.7`
- `BM25_WEIGHT`: `0.3`
- `CANDIDATE_K`: `20`
- `TOP_K`: `5`
- `RERANKER_ENABLED`: `False`

## Empirical Retrieval Quality & Latency Summary

### Quality Evaluation (Recall & MRR)
| Method / Weights | Recall@1 | Recall@5 | Recall@10 | MRR@5 | MRR@10 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dense Only (1.0/0.0)** | 0.1667 | 0.6154 | 0.7821 | 0.3427 | 0.3636 |
| **BM25 Only (0.0/1.0)** | 0.2051 | 0.6154 | 0.7308 | 0.3694 | 0.3852 |
| **Hybrid (0.9/0.1)** | 0.1923 | 0.6282 | **0.8333** | 0.3641 | 0.3903 |
| **Hybrid (0.8/0.2)** | 0.2051 | 0.6282 | **0.8333** | 0.3716 | 0.4003 |
| **Production Hybrid (0.7/0.3)** | **0.2051** | **0.7051** | 0.8205 | **0.3902** | **0.4056** |
| **Hybrid (0.6/0.4)** | **0.2051** | **0.7179** | 0.8205 | **0.3987** | **0.4133** |
| **Hybrid + Reranker** | 0.1200 | 0.4000 | 0.5600 | 0.2180 | 0.2388 |
