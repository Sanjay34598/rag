import os
from typing import List, Dict, Any
from app.core.config import settings

class RerankerService:
    _instance = None

    def __new__(cls, model_name: str = None):
        if cls._instance is None:
            cls._instance = super(RerankerService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = None):
        if self._initialized:
            return
        
        os.environ["USE_TF"] = "0"
        from sentence_transformers import CrossEncoder
        self.model_name = model_name or settings.RERANKER_MODEL
        print(f"[RerankerService] Initializing cross-encoder model: {self.model_name}")
        self.model = CrossEncoder(self.model_name)
        self._initialized = True
        print(f"[RerankerService] Cross-encoder model loaded successfully.")

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        pairs = [[query, cand["text"]] for cand in candidates]
        scores = self.model.predict(pairs)

        reranked = []
        for cand, score in zip(candidates, scores):
            item = dict(cand)
            item["rerank_score"] = float(score)
            item["score"] = float(score)  # Update main score to rerank score
            reranked.append(item)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

def get_reranker_service() -> RerankerService:
    return RerankerService()
