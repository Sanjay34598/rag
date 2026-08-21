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

from app.services.retrieval.vitamin_expansion import expand_vitamin_query, does_chunk_support_intent

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

    def ensure_loaded(self):
        if not getattr(self.vector_store, "_is_loaded", False) or not getattr(self.bm25_retriever, "_is_loaded", False):
            self.load()

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
        self.embedding_service = get_embedding_service()
        self.query_alignment_service = get_query_alignment_service()
        
        canonical_dir = getattr(settings, "CANONICAL_INDEX_DIR", DATA_DIR / "indexes" / "canonical")
        if not canonical_dir.exists():
            canonical_dir = DATA_DIR / "indexes" / "canonical"

        if not canonical_dir.exists():
            raise FileNotFoundError(
                f"Canonical index directory missing at {canonical_dir}. Fallback to legacy index is strictly forbidden."
            )

        self.canonical_retriever = CanonicalRetriever(index_dir=canonical_dir)
        if load_indexes:
            print("[RAG INIT] Loading canonical FAISS index...")
            print("[RAG INIT] Loading canonical BM25 index...")
            print("[RAG INIT] Loading processed chunks...")
            self.canonical_retriever.load()
            self.query_alignment_service.initialize()
            print("[RAG INIT] Pre-warming complete")
            
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
            self.initialize(load_indexes=False)

        self.canonical_retriever.ensure_loaded()

        target_lang = normalize_language_code(language_code)
        actual_top_k = top_k if top_k is not None else settings.TOP_K
        actual_candidate_k = candidate_k if candidate_k is not None else settings.CANDIDATE_K
        use_reranker = reranker_enabled if reranker_enabled is not None else settings.RERANKER_ENABLED

        start_total = time.perf_counter()

        # 0. Vitamin Terminology & Intent Expansion Check
        expansion_info = expand_vitamin_query(query)

        # 1. Dataset Query Alignment (Stage 1)
        if self.query_alignment_service is None:
            self.query_alignment_service = get_query_alignment_service()

        alignment_info = self.query_alignment_service.align(query)
        bm25_query = query
        if alignment_info["matched"]:
            bm25_query = alignment_info["aligned_english_query"]

        # 2. Dense Vector Search & BM25 Search (Pass 1)
        t0 = time.perf_counter()
        query_vector = self.embedding_service.encode_query(query)
        t_embed = (time.perf_counter() - t0) * 1000.0
        
        t0 = time.perf_counter()
        dense_results = self.canonical_retriever.vector_store.search(query_vector, top_k=actual_candidate_k)
        t_faiss = (time.perf_counter() - t0) * 1000.0
        
        t0 = time.perf_counter()
        bm25_results = self.canonical_retriever.bm25_retriever.search(bm25_query, top_k=actual_candidate_k)
        t_bm25 = (time.perf_counter() - t0) * 1000.0
        
        # 4. Hybrid Score Fusion (Pass 1)
        t0 = time.perf_counter()
        hybrid_candidates = self.canonical_retriever.hybrid_retriever.fuse(
            dense_results=dense_results,
            bm25_results=bm25_results,
            top_k=actual_candidate_k
        )
        t_fusion = (time.perf_counter() - t0) * 1000.0

        top_pass1_score = hybrid_candidates[0]["score"] if hybrid_candidates else 0.0
        used_pass2 = False

        # Check if Pass 1 retrieved top candidates containing target vitamin terms
        pass1_has_target = False
        if hybrid_candidates and expansion_info["has_vitamin"]:
            v_keys = expansion_info.get("vitamin_keys", [])
            syns = set()
            from app.services.retrieval.vitamin_expansion import VITAMIN_SYNONYMS
            for vk in v_keys:
                syns.update(VITAMIN_SYNONYMS.get(vk, []))
            for cand in hybrid_candidates[:3]:
                cand_text = cand.get("text", "").lower()
                if any(syn in cand_text for syn in syns):
                    pass1_has_target = True
                    break

        # 4b. Controlled Pass 2 (Query Expansion Fallback if Pass 1 top score is low OR Pass 1 candidates lack target vitamin terms)
        if expansion_info["has_vitamin"] and (top_pass1_score < getattr(settings, "MIN_RETRIEVAL_SCORE", 0.20) or not pass1_has_target):
            used_pass2 = True
            all_candidate_map: Dict[str, Dict[str, Any]] = {c["chunk_id"]: c for c in hybrid_candidates}
            
            for exp_q in expansion_info["expanded_queries"]:
                exp_vec = self.embedding_service.encode_query(exp_q)
                exp_dense = self.canonical_retriever.vector_store.search(exp_vec, top_k=actual_candidate_k)
                exp_bm25 = self.canonical_retriever.bm25_retriever.search(exp_q, top_k=actual_candidate_k)
                exp_fused = self.canonical_retriever.hybrid_retriever.fuse(
                    dense_results=exp_dense,
                    bm25_results=exp_bm25,
                    top_k=actual_candidate_k
                )
                for cand in exp_fused:
                    cid = cand["chunk_id"]
                    if cid not in all_candidate_map or cand["score"] > all_candidate_map[cid]["score"]:
                        all_candidate_map[cid] = cand

            hybrid_candidates = list(all_candidate_map.values())

        # Terminology boost for candidates matching specific vitamin terms
        if expansion_info["has_vitamin"]:
            v_keys = expansion_info.get("vitamin_keys", [])
            syns = set()
            from app.services.retrieval.vitamin_expansion import VITAMIN_SYNONYMS
            for vk in v_keys:
                syns.update(VITAMIN_SYNONYMS.get(vk, []))
            
            specific_terms = []
            for s in syns:
                if len(s) >= 4 or s.startswith("b-") or " " in s:
                    specific_terms.append(s)
                elif s not in ("b1", "b2", "b3", "b5", "b6", "b7", "b9", "b12"):
                    specific_terms.append(f"vitamin {s}")
                else:
                    specific_terms.append(f"vitamin {s}")
                    specific_terms.append(f"vitamin {s[0]}-{s[1:]}")
                    specific_terms.append(f"b-{s[1:]}")

            for cand in hybrid_candidates:
                cand_text = cand.get("text", "").lower()
                matched = any(re.search(r'\b' + re.escape(s) + r'\b', cand_text) for s in specific_terms)
                if matched:
                    cand["score"] = cand.get("score", 0.0) + 2.0
                elif "vitamin" in cand_text and not matched:
                    cand["score"] = cand.get("score", 0.0) * 0.5

        hybrid_candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        hybrid_candidates = hybrid_candidates[:actual_candidate_k]
        
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

        print("[RETRIEVAL DIAGNOSTIC]")
        print(f"original_query='{query}'")
        print(f"normalized_query='{expansion_info['normalized_query']}'")
        print(f"expanded_queries={expansion_info['expanded_queries']}")
        print(f"query_intent='{expansion_info['intent']}'")
        print(f"retrieval_pass={'pass_2_expansion' if used_pass2 else 'pass_1_direct'}")
        print(f"dense_score={top_dense}")
        print(f"bm25_score={top_bm25}")
        print(f"final_score={top_final}")

        return {
            "query": query,
            "language": target_lang,
            "results": clean_results,
            "expansion_info": expansion_info,
            "used_pass2": used_pass2,
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

