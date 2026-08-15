# Stage 5A RAG Pipeline, Guardrails & Grounding Documentation

## RAG Pipeline Architecture
The end-to-end RAG system orchestrates input validation, hybrid retrieval, confidence checks, context formatting, LLM answer generation, and grounding validation.

```
User Query
    ↓
1. Input Guardrail (Length & empty query check)
    ↓
2. Hybrid Retrieval Service (FAISS + BM25, Reranker Disabled)
    ↓
3. Retrieval Guardrail (Raw confidence check)
    ↓
4. Prompt Injection Guardrail (Untrusted content sanitization)
    ↓
5. Context Builder & Prompt Builder
    ↓
6. Answer Generator (Groq API or Extracted Offline Fallback)
    ↓
7. Output Guardrail (Content safety & refusal filter)
    ↓
8. Grounding Validator (Stopword-filtered token overlap check)
    ↓
Final RAG Response Output
```

## LLM Configuration & Modes
- **Configured Provider**: `groq`
- **Configured Model**: `llama-3.1-8b-instant`
- **Current Mode**: `fallback` (No external API key set in current local environment)
- **Status Flag**: `REAL_LLM_CONFIGURATION = NOT AVAILABLE`

## Guardrails & Validation Behavior
1. **Input Guardrail**: Rejects empty strings, whitespace, or excessively long queries before hitting retrieval.
2. **Retrieval Guardrail**: Evaluates raw dense cosine similarity and BM25 scores. If confidence is insufficient (`dense < 0.50` and `bm25 < 1.0`), execution halts with a polite refusal without wasting LLM calls.
3. **Prompt Injection Guardrail**: Wraps retrieved passage text in `[UNTRUSTED_CONTENT_FILTERED]` tags to prevent prompt hijacking.
4. **Grounding Validator**: Filters Devanagari & English stopwords to compute token overlap between generated answers and retrieved context. Answers with overlap ratio `< 0.35` are rejected (`grounded = false`).

## Empirical Performance & Latency Breakdown (Fallback Mode - 10 Queries)

| Pipeline Stage | Average (ms) | P50 (ms) | P100 (ms) |
| :--- | :--- | :--- | :--- |
| **Retrieval Stage** | 40.51 ms | 37.98 ms | 61.02 ms |
| **Context & Prompt Formatting** | 0.33 ms | 0.22 ms | 1.31 ms |
| **LLM Generation / Fallback** | 0.01 ms | 0.01 ms | 0.02 ms |
| **Grounding Validation** | 0.29 ms | 0.30 ms | 0.36 ms |
| **TOTAL END-TO-END RAG LATENCY** | **41.17 ms** | **38.52 ms** | **61.67 ms** |
