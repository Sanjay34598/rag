import os
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from app.core.config import settings, DATA_DIR
from app.services.retrieval.embeddings import get_embedding_service, EmbeddingService
from app.services.retrieval.vector_store import FAISSVectorStore
from app.services.retrieval.bm25_retriever import BM25Retriever
from app.services.retrieval.hybrid_retriever import HybridRetriever
from app.services.retrieval.reranker import get_reranker_service, RerankerService
from app.services.retrieval.query_alignment import get_query_alignment_service, QueryAlignmentService

def normalize_language_code(lang_code: Optional[str]) -> str:
    if not lang_code:
        return "en"
    code = lang_code.strip().lower()
    if code in ("en-in", "en_in", "en", "english"):
        return "en"
    if code in ("te-in", "te_in", "te", "telugu"):
        return "te"
    if code in ("hi-in", "hi_in", "hi", "hindi"):
        return "hi"
    return "en"

class CanonicalRetriever:
    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.vector_store = FAISSVectorStore()
        self.vector_store.index_path = str(index_dir / "faiss_index.bin")
        self.vector_store.metadata_path = str(index_dir / "faiss_metadata.json")
        
        self.bm25_retriever = BM25Retriever()
        self.bm25_retriever.index_path = str(index_dir / "bm25_index.pkl")
        self.hybrid_retriever = HybridRetriever()

    def load(self):
        faiss_path = self.index_dir / "faiss_index.bin"
        bm25_path = self.index_dir / "bm25_index.pkl"
        chunks_path = self.index_dir / "processed_chunks.json"

        if not faiss_path.exists() or not bm25_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"Strict load requirement failed: Canonical index is incomplete at {self.index_dir}. "
                f"Required files: faiss_index.bin, bm25_index.pkl, processed_chunks.json."
            )
        self.vector_store.load()
        self.bm25_retriever.load()
        print(f"[CanonicalRetriever] Pre-warmed canonical index loaded successfully ({len(self.vector_store.metadata)} chunks from {self.index_dir}).")

class RetrievalService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RetrievalService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.embedding_service: Optional[EmbeddingService] = None
        self.canonical_retriever: Optional[CanonicalRetriever] = None
        self.retrievers: Dict[str, CanonicalRetriever] = {}  # for backward compatibility
        self.reranker_service: Optional[RerankerService] = None
        self.query_alignment_service: Optional[QueryAlignmentService] = None
        self._initialized = True

    def initialize(self, load_indexes: bool = True) -> None:
        print("[RetrievalService] Initializing pre-warmed canonical retrieval index...")
        self.embedding_service = get_embedding_service()
        self.query_alignment_service = get_query_alignment_service()
        self.query_alignment_service.initialize()
        
        canonical_dir = getattr(settings, "CANONICAL_INDEX_DIR", DATA_DIR / "indexes" / "canonical")
        if not canonical_dir.exists():
            canonical_dir = DATA_DIR / "indexes" / "canonical"

        if not canonical_dir.exists():
            raise FileNotFoundError(
                f"Canonical index directory missing at {canonical_dir}. Fallback to legacy index is strictly forbidden."
            )

        self.canonical_retriever = CanonicalRetriever(index_dir=canonical_dir)
        if load_indexes:
            self.canonical_retriever.load()
            
        for l_key in ["en", "hi", "te", "canonical"]:
            self.retrievers[l_key] = self.canonical_retriever

        if settings.RERANKER_ENABLED:
            self.reranker_service = get_reranker_service()
            
        print(f"[RetrievalService] Canonical index initialization complete at {canonical_dir}.")

    def retrieve(
        self,
        query: str,
        language_code: Optional[str] = None,
        top_k: Optional[int] = None,
        candidate_k: Optional[int] = None,
        reranker_enabled: Optional[bool] = None
    ) -> Dict[str, Any]:
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace-only.")

        if not self.canonical_retriever:
            self.initialize(load_indexes=True)

        target_lang = normalize_language_code(language_code)
        actual_top_k = top_k if top_k is not None else settings.TOP_K
        actual_candidate_k = candidate_k if candidate_k is not None else settings.CANDIDATE_K
        use_reranker = reranker_enabled if reranker_enabled is not None else settings.RERANKER_ENABLED

        start_total = time.perf_counter()

        # 1. Dataset Query Alignment (Stage 1)
        if self.query_alignment_service is None:
            self.query_alignment_service = get_query_alignment_service()

        alignment_info = self.query_alignment_service.align(query)
        bm25_query = query
        if alignment_info["matched"]:
            bm25_query = alignment_info["aligned_english_query"]

        # 2. Dense Vector Search (Multilingual Embedding maps EN, HI, TE queries into canonical vector space)
        t0 = time.perf_counter()
        query_vector = self.embedding_service.encode_query(query)
        t_embed = (time.perf_counter() - t0) * 1000.0
        
        t0 = time.perf_counter()
        dense_results = self.canonical_retriever.vector_store.search(query_vector, top_k=actual_candidate_k)
        t_faiss = (time.perf_counter() - t0) * 1000.0
        
        # 3. BM25 Keyword Search
        t0 = time.perf_counter()
        bm25_results = self.canonical_retriever.bm25_retriever.search(bm25_query, top_k=actual_candidate_k)
        t_bm25 = (time.perf_counter() - t0) * 1000.0
        
        # 4. Hybrid Score Fusion
        t0 = time.perf_counter()
        hybrid_candidates = self.canonical_retriever.hybrid_retriever.fuse(
            dense_results=dense_results,
            bm25_results=bm25_results,
            top_k=actual_candidate_k
        )
        t_fusion = (time.perf_counter() - t0) * 1000.0
        
        # 5. Lightweight Reranking (Optional)
        t0 = time.perf_counter()
        if use_reranker:
            if self.reranker_service is None:
                self.reranker_service = get_reranker_service()
            final_results = self.reranker_service.rerank(
                query=query,
                candidates=hybrid_candidates,
                top_k=actual_top_k
            )
        else:
            final_results = hybrid_candidates[:actual_top_k]
        t_rerank = (time.perf_counter() - t0) * 1000.0
        
        total_latency = (time.perf_counter() - start_total) * 1000.0
        
        # Format clean canonical results structure
        clean_results = []
        top_dense = 0.0
        top_bm25 = 0.0
        top_final = 0.0

        for idx, r in enumerate(final_results):
            meta = r.get("metadata", {})
            chunk_id = r.get("chunk_id", "unknown")
            text = r.get("text", "")
            sc = round(float(r.get("score", 0.0)), 4)
            d_sc = round(float(r.get("dense_score", 0.0)), 4)
            b_sc = round(float(r.get("bm25_score", 0.0)), 4)

            if idx == 0:
                top_dense = d_sc
                top_bm25 = b_sc
                top_final = sc

            clean_results.append({
                "chunk_id": chunk_id,
                "language": "en",  # Canonical evidence language is ALWAYS English
                "query_id": r.get("query_id") or meta.get("query_id", 0),
                "text": text,
                "score": sc,
                "dense_score": d_sc,
                "bm25_score": b_sc,
                "source_lang": meta.get("source_lang", "eng_Latn"),
                "target_lang": None,
                "source_type": meta.get("source_type", "original_english"),
                "metadata": meta
            })

        print("[RETRIEVAL]")
        print(f"input_language={language_code or 'en-IN'}")
        print(f"retrieval_language=en")
        print(f"query_alignment={alignment_info['query_alignment']}")
        print(f"canonical_query_id={alignment_info['canonical_query_id']}")
        print(f"alignment_score={alignment_info['alignment_score']}")
        print(f"dense_score={top_dense}")
        print(f"bm25_score={top_bm25}")
        print(f"final_score={top_final}")

        return {
            "query": query,
            "language": target_lang,
            "results": clean_results,
            "reranking_enabled": use_reranker,
            "latency_ms": round(total_latency, 2),
            "latency_breakdown": {
                "embedding_ms": round(t_embed, 2),
                "faiss_ms": round(t_faiss, 2),
                "bm25_ms": round(t_bm25, 2),
                "fusion_ms": round(t_fusion, 2),
                "reranking_ms": round(t_rerank, 2)
            }
        }

def get_retrieval_service() -> RetrievalService:
    return RetrievalService()
