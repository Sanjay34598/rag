import time
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.services.retrieval.embeddings import get_embedding_service, EmbeddingService
from app.services.retrieval.vector_store import FAISSVectorStore
from app.services.retrieval.bm25_retriever import BM25Retriever
from app.services.retrieval.hybrid_retriever import HybridRetriever
from app.services.retrieval.reranker import get_reranker_service, RerankerService

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
        self.vector_store: Optional[FAISSVectorStore] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.hybrid_retriever: Optional[HybridRetriever] = None
        self.reranker_service: Optional[RerankerService] = None
        self._initialized = True

    def initialize(
        self,
        vector_store: Optional[FAISSVectorStore] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        load_indexes: bool = True
    ) -> None:
        print("[RetrievalService] Initializing retrieval models and indexes...")
        self.embedding_service = get_embedding_service()
        
        if vector_store is None:
            self.vector_store = FAISSVectorStore()
            if load_indexes:
                self.vector_store.load()
        else:
            self.vector_store = vector_store
            
        if bm25_retriever is None:
            self.bm25_retriever = BM25Retriever()
            if load_indexes:
                self.bm25_retriever.load()
        else:
            self.bm25_retriever = bm25_retriever
            
        self.hybrid_retriever = HybridRetriever()
        
        if settings.RERANKER_ENABLED:
            self.reranker_service = get_reranker_service()
            
        print("[RetrievalService] Initialization complete.")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        candidate_k: Optional[int] = None,
        reranker_enabled: Optional[bool] = None
    ) -> Dict[str, Any]:
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace-only.")

        if self.embedding_service is None:
            self.initialize(load_indexes=True)

        actual_top_k = top_k if top_k is not None else settings.TOP_K
        actual_candidate_k = candidate_k if candidate_k is not None else settings.CANDIDATE_K
        use_reranker = reranker_enabled if reranker_enabled is not None else settings.RERANKER_ENABLED

        start_total = time.perf_counter()
        
        # 1. Query Embedding
        t0 = time.perf_counter()
        query_vector = self.embedding_service.encode_query(query)
        t_embed = (time.perf_counter() - t0) * 1000.0
        
        # 2. FAISS Vector Search
        t0 = time.perf_counter()
        dense_results = self.vector_store.search(query_vector, top_k=actual_candidate_k)
        t_faiss = (time.perf_counter() - t0) * 1000.0
        
        # 3. BM25 Keyword Search
        t0 = time.perf_counter()
        bm25_results = self.bm25_retriever.search(query, top_k=actual_candidate_k)
        t_bm25 = (time.perf_counter() - t0) * 1000.0
        
        # 4. Hybrid Score Fusion
        t0 = time.perf_counter()
        hybrid_candidates = self.hybrid_retriever.fuse(
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
        
        # Clean results structure
        clean_results = []
        for r in final_results:
            clean_results.append({
                "chunk_id": r["chunk_id"],
                "text": r["text"],
                "score": round(float(r.get("score", 0.0)), 4),
                "dense_score": round(float(r.get("dense_score", 0.0)), 4),
                "bm25_score": round(float(r.get("bm25_score", 0.0)), 4),
                "metadata": r.get("metadata", {})
            })

        return {
            "query": query,
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
