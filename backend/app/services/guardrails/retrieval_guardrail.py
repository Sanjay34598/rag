from typing import List, Dict, Any, Tuple
from app.core.config import settings

class RetrievalGuardrail:
    def __init__(self, min_score: float = None, min_chunks: int = None):
        self.min_score = min_score if min_score is not None else settings.MIN_RETRIEVAL_SCORE
        self.min_chunks = min_chunks if min_chunks is not None else settings.MIN_CONTEXT_CHUNKS

    def evaluate(self, chunks: List[Dict[str, Any]]) -> Tuple[bool, float, str]:
        REFUSAL_MSG = "I couldn't find enough relevant information in the available knowledge base to answer that question."

        if not chunks or len(chunks) < self.min_chunks:
            return False, 0.0, REFUSAL_MSG

        top_chunk = chunks[0]
        top_score = float(top_chunk.get("score", 0.0))
        top_dense = float(top_chunk.get("dense_score", top_score))
        top_bm25 = float(top_chunk.get("bm25_score", 0.0))
        
        if top_score < self.min_score or (top_dense < 0.22 and top_bm25 < 3.0 and top_score < 0.5):
            return False, top_score, REFUSAL_MSG

        return True, top_score, ""
