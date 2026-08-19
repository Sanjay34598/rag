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
        self.model_name = model_name or settings.RERANKER_MODEL
        self._model = None
        self._initialized = True

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            print(f"[RerankerService] Lazy loading cross-encoder model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
            print(f"[RerankerService] Cross-encoder model loaded successfully.")
        return self._model

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
