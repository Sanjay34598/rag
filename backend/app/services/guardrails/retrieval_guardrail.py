from typing import List, Dict, Any, Tuple
from app.core.config import settings

class RetrievalGuardrail:
    def __init__(self, min_score: float = None, min_chunks: int = None):
        self.min_score = min_score if min_score is not None else settings.MIN_RETRIEVAL_SCORE
        self.min_chunks = min_chunks if min_chunks is not None else settings.MIN_CONTEXT_CHUNKS

    def evaluate(self, chunks: List[Dict[str, Any]]) -> Tuple[bool, float, str]:
        if not chunks or len(chunks) < self.min_chunks:
            return False, 0.0, "I couldn't find enough relevant information in the available knowledge base to answer that question."

        top_score = max((c.get("score", 0.0) for c in chunks), default=0.0)
        top_dense = max((c.get("dense_score", top_score) for c in chunks), default=0.0)
        top_bm25 = max((c.get("bm25_score", 0.0) for c in chunks), default=0.0)
        
        # Consider confidence sufficient if top_score >= min_score and (dense >= min_score or bm25 >= 1.0)
        if top_score < self.min_score or (top_dense < self.min_score and top_bm25 < 1.0):
            return False, float(top_score), "I couldn't find sufficiently relevant information in the available knowledge base to answer that question."

        return True, float(top_score), ""
