from typing import List, Dict, Any, Tuple
from app.core.config import settings

class RetrievalGuardrail:
    def __init__(self, min_score: float = None, min_chunks: int = None):
        self.min_score = min_score if min_score is not None else settings.MIN_RETRIEVAL_SCORE
        self.min_chunks = min_chunks if min_chunks is not None else settings.MIN_CONTEXT_CHUNKS

    def evaluate(self, chunks: List[Dict[str, Any]], query: str = "", language_code: str = None) -> Tuple[bool, float, str]:
        if language_code == "te-IN" or language_code == "te":
            refusal_msg = "అందుబాటులో ఉన్న సమాచారం ఆధారంగా ఈ సమాధానాన్ని ధృవీకరించలేకపోయాను."
        elif language_code == "hi-IN" or language_code == "hi":
            refusal_msg = "उपलब्ध संदर्भ से मैं इस उत्तर की पुष्टि नहीं कर सका।"
        else:
            refusal_msg = "I couldn't verify that answer from the available context."

        if not chunks or len(chunks) < self.min_chunks:
            return False, 0.0, refusal_msg

        top_chunk = chunks[0]
        top_score = float(top_chunk.get("score", 0.0))
        top_dense = float(top_chunk.get("dense_score", top_score))
        top_bm25 = float(top_chunk.get("bm25_score", 0.0))
        
        # Check if top chunk meets minimum relevance threshold
        if top_score < self.min_score or (top_dense < 0.22 and top_bm25 < 3.0 and top_score < 0.5):
            return False, top_score, refusal_msg

        return True, top_score, ""
