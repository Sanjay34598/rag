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

def normalize_language_code(lang_code: Optional[str]) -> str:
    if not lang_code:
        return "hi"
    code = lang_code.strip().lower()
    if code in ("en-in", "en_in", "en", "english"):
        return "en"
    if code in ("te-in", "te_in", "te", "telugu"):
        return "te"
    if code in ("hi-in", "hi_in", "hi", "hindi"):
        return "hi"
    return "hi"

def validate_source_script(target_lang: str, chunk_id: str, text: str) -> None:
    has_devanagari = bool(re.search(r'[\u0900-\u097F]', text))
    has_telugu = bool(re.search(r'[\u0C00-\u0C7F]', text))

    if target_lang == "en":
        if has_devanagari or has_telugu:
            raise ValueError(
                f"Corrupted retrieval: English index returned non-English script source in chunk '{chunk_id}'. "
                f"Script detected: Devanagari={has_devanagari}, Telugu={has_telugu}."
            )
    elif target_lang == "hi":
        if has_telugu:
            raise ValueError(
                f"Corrupted retrieval: Hindi index returned Telugu script source in chunk '{chunk_id}'."
            )
    elif target_lang == "te":
        if has_devanagari:
            raise ValueError(
                f"Corrupted retrieval: Telugu index returned Hindi script source in chunk '{chunk_id}'."
            )

class SingleLanguageRetriever:
    def __init__(self, language: str, index_dir: Path):
        self.language = language
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
                f"Strict load requirement failed: Language index for '{self.language}' is incomplete at {self.index_dir}. "
                f"Required files: faiss_index.bin, bm25_index.pkl, processed_chunks.json."
            )
        self.vector_store.load()
        self.bm25_retriever.load()
        print(f"[SingleLanguageRetriever] Pre-warmed index for '{self.language}' loaded successfully ({len(self.vector_store.metadata)} chunks from {self.index_dir}).")

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
        self.retrievers: Dict[str, SingleLanguageRetriever] = {}
        self.reranker_service: Optional[RerankerService] = None
        self._initialized = True

    def initialize(self, load_indexes: bool = True) -> None:
        print("[RetrievalService] Initializing pre-warmed multilingual retrieval indexes...")
        self.embedding_service = get_embedding_service()
        indexes_base_dir = getattr(settings, "INDEXES_DIR", DATA_DIR / "indexes")

        for lang in ["en", "hi", "te"]:
            lang_dir = indexes_base_dir / lang
            if not lang_dir.exists():
                raise FileNotFoundError(
                    f"Multilingual index directory missing for language '{lang}' at {lang_dir}. Fallback to root DATA_DIR is strictly forbidden."
                )

            retriever = SingleLanguageRetriever(language=lang, index_dir=lang_dir)
            if load_indexes:
                retriever.load()
            self.retrievers[lang] = retriever

        if settings.RERANKER_ENABLED:
            self.reranker_service = get_reranker_service()
            
        print(f"[RetrievalService] Initialization complete. Active pre-warmed language retrievers: {list(self.retrievers.keys())}")

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

        if not self.retrievers:
            self.initialize(load_indexes=True)

        target_lang = normalize_language_code(language_code)
        
        # NEVER SILENTLY FALL BACK TO HINDI IF REQUESTED LANGUAGE IS MISSING
        if target_lang not in self.retrievers:
            raise ValueError(f"Language index for '{target_lang}' (requested '{language_code}') is unavailable on backend.")

        lang_retriever = self.retrievers[target_lang]

        actual_top_k = top_k if top_k is not None else settings.TOP_K
        actual_candidate_k = candidate_k if candidate_k is not None else settings.CANDIDATE_K
        use_reranker = reranker_enabled if reranker_enabled is not None else settings.RERANKER_ENABLED

        start_total = time.perf_counter()
        
        # 1. Query Embedding
        t0 = time.perf_counter()
        query_vector = self.embedding_service.encode_query(query)
        t_embed = (time.perf_counter() - t0) * 1000.0
        
        # 2. FAISS Vector Search on Language-Specific Index
        t0 = time.perf_counter()
        dense_results = lang_retriever.vector_store.search(query_vector, top_k=actual_candidate_k)
        t_faiss = (time.perf_counter() - t0) * 1000.0
        
        # 3. BM25 Keyword Search on Language-Specific Index
        t0 = time.perf_counter()
        bm25_results = lang_retriever.bm25_retriever.search(query, top_k=actual_candidate_k)
        t_bm25 = (time.perf_counter() - t0) * 1000.0
        
        # 4. Hybrid Score Fusion
        t0 = time.perf_counter()
        hybrid_candidates = lang_retriever.hybrid_retriever.fuse(
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
        
        # Clean results structure with metadata & DUAL METADATA + SCRIPT TEXT VALIDATION
        clean_results = []
        for r in final_results:
            meta = r.get("metadata", {})
            chunk_lang = r.get("language") or meta.get("language") or target_lang
            chunk_id = r.get("chunk_id", "unknown")
            text = r.get("text", "")

            # 1. Metadata language check
            if chunk_lang != target_lang:
                raise ValueError(
                    f"Metadata language mismatch in retrieval results: requested '{target_lang}' but chunk '{chunk_id}' had language '{chunk_lang}'."
                )

            # 2. Actual script text content validation (Step 5)
            validate_source_script(target_lang=target_lang, chunk_id=chunk_id, text=text)

            clean_results.append({
                "chunk_id": chunk_id,
                "language": target_lang,
                "query_id": r.get("query_id") or meta.get("query_id", 0),
                "text": text,
                "score": round(float(r.get("score", 0.0)), 4),
                "dense_score": round(float(r.get("dense_score", 0.0)), 4),
                "bm25_score": round(float(r.get("bm25_score", 0.0)), 4),
                "source_lang": meta.get("source_lang", "eng_Latn"),
                "target_lang": meta.get("target_lang", "hin_Deva"),
                "metadata": meta
            })

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
