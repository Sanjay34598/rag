from typing import List, Dict, Any, Tuple
from app.core.config import settings
from app.services.retrieval.vitamin_expansion import extract_query_intent, does_chunk_support_intent

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

        if not chunks:
            return False, 0.0, refusal_msg

        query_intent = extract_query_intent(query)
        intent_aligned_chunks = []
        for c in chunks:
            supports_intent, intent_reason = does_chunk_support_intent(c, query_intent, query=query)
            print(f"[INTENT ALIGNMENT DIAGNOSTIC] query='{query}' | intent='{query_intent}' | chunk={c.get('chunk_id')} | supports_intent={supports_intent} | reason='{intent_reason}'")
            if supports_intent:
                intent_aligned_chunks.append(c)

        if not intent_aligned_chunks or len(intent_aligned_chunks) < self.min_chunks:
            print(f"[RETRIEVAL GUARDRAIL REJECTION] No valid intent-aligned chunks remaining for query intent '{query_intent}'.")
            return False, 0.0, refusal_msg

        top_chunk = intent_aligned_chunks[0]
        top_score = float(top_chunk.get("score", 0.0))
        top_dense = float(top_chunk.get("dense_score", top_score))
        top_bm25 = float(top_chunk.get("bm25_score", 0.0))
        
        # Check if top chunk meets minimum relevance threshold
        if top_score < self.min_score or (top_dense < 0.45 and top_bm25 < 1.0) or (top_dense < 0.22 and top_score < 0.5):
            print(f"[RETRIEVAL GUARDRAIL] Rejected top chunk {top_chunk.get('chunk_id')}: score {top_score:.4f} < min_score {self.min_score}")
            return False, top_score, refusal_msg

        return True, top_score, ""

