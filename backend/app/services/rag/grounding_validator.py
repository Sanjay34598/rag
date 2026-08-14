import re
from typing import List, Dict, Any, Tuple

HINDI_ENGLISH_STOPWORDS = {
    "का", "के", "की", "है", "हैं", "और", "में", "से", "पर", "को", "ने", "या", "कि", "यह", "वह", "जो", "यहाँ", "वहाँ", "स्थित",
    "the", "is", "are", "was", "were", "of", "and", "in", "to", "a", "an", "or", "for", "on", "with", "at", "by", "from"
}

def tokenize_text(text: str) -> set:
    if not text:
        return set()
    text = text.lower()
    tokens = re.findall(r'[\w\u0900-\u097F]+', text)
    # Return non-empty alphanumeric / Devanagari word tokens excluding stopwords
    return {t for t in tokens if len(t) >= 2 and t not in HINDI_ENGLISH_STOPWORDS}

class GroundingValidator:
    def __init__(self, min_token_overlap_ratio: float = 0.35):
        self.min_token_overlap_ratio = min_token_overlap_ratio

    def validate(self, answer: str, context_chunks: List[Dict[str, Any]]) -> Tuple[bool, float, str]:
        if not answer or not answer.strip():
            return False, 0.0, "Generated answer was empty."

        if not context_chunks:
            return False, 0.0, "No context chunks were provided for grounding validation."

        answer_tokens = tokenize_text(answer)
        if not answer_tokens:
            return False, 0.0, "Answer contained no valid tokens."

        context_text = " ".join([c.get("text", "") for c in context_chunks])
        context_tokens = tokenize_text(context_text)

        if not context_tokens:
            return False, 0.0, "Context contained no valid tokens."

        # Calculate token overlap ratio
        overlap = answer_tokens.intersection(context_tokens)
        overlap_ratio = len(overlap) / float(len(answer_tokens))

        if overlap_ratio < self.min_token_overlap_ratio:
            return False, round(overlap_ratio, 2), "I couldn't verify that answer from the available context."

        confidence = min(1.0, round(0.5 + overlap_ratio * 0.5, 2))
        return True, confidence, answer
