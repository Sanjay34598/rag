import re
from typing import List, Dict, Any, Tuple

HINDI_ENGLISH_TELUGU_STOPWORDS = {
    "का", "के", "की", "है", "हैं", "और", "में", "से", "पर", "को", "ने", "या", "कि", "यह", "वह", "जो", "यहाँ", "वहाँ", "स्थित",
    "the", "is", "are", "was", "were", "of", "and", "in", "to", "a", "an", "or", "for", "on", "with", "at", "by", "from"
}

def tokenize_text(text: str) -> set:
    if not text:
        return set()
    text = text.lower()
    tokens = re.findall(r'[\w\u0900-\u097F\u0C00-\u0C7F]+', text)
    # Return non-empty word tokens excluding common stopwords
    return {t for t in tokens if len(t) >= 2 and t not in HINDI_ENGLISH_TELUGU_STOPWORDS}

class GroundingValidator:
    def __init__(self, min_token_overlap_ratio: float = 0.35):
        self.min_token_overlap_ratio = min_token_overlap_ratio

    def validate(self, answer: str, context_chunks: List[Dict[str, Any]], query: str = "", language_code: str = None) -> Tuple[bool, float, str]:
        REFUSAL_MSG = "I couldn't verify that answer from the available context."

        if not answer or not answer.strip():
            return False, 0.0, "Generated answer was empty."

        # Check refusal strings across supported languages
        refusal_triggers = [
            "पर्याप्त जानकारी नहीं मिली", "couldn't find", "couldn't verify",
            "Refused:", "क्षमा करें", "క్షమించండి", "సమాధానం రూపొందించడానికి", "temporarily unavailable"
        ]
        if any(trigger in answer for trigger in refusal_triggers):
            return False, 0.0, answer

        if not context_chunks:
            return False, 0.0, REFUSAL_MSG

        # Conversational / courtesy phrase filter
        if query:
            clean_query = query.strip().lower()
            conversational_phrases = {
                "आपके लिए धन्यवाद", "धन्यवाद", "शुक्रिया", "थैंक यू", "नमस्ते", 
                "hello", "hi", "thank you", "thanks", "thank you very much",
                "నమస్కారం", "ధన్యవాదాలు", "వీడ్కోలు"
            }
            if clean_query in conversational_phrases:
                return False, 0.0, REFUSAL_MSG

        # For cross-lingual answers (e.g. Telugu/English answer generated from Hindi context):
        # String token overlap will naturally be near 0 because script/vocabulary differs.
        # If Groq generated a valid answer and top retrieval chunk score >= threshold, mark grounded.
        if language_code in ("te-IN", "en-IN"):
            top_score = float(context_chunks[0].get("score", 0.0)) if context_chunks else 0.0
            if top_score >= 0.20 and len(answer.strip()) > 10:
                conf = min(1.0, round(0.5 + min(1.0, top_score) * 0.5, 2))
                return True, conf, answer

        answer_tokens = tokenize_text(answer)
        if not answer_tokens:
            return False, 0.0, "Answer contained no valid tokens."

        context_text = " ".join([c.get("text", "") for c in context_chunks])
        context_tokens = tokenize_text(context_text)

        if not context_tokens:
            return False, 0.0, REFUSAL_MSG

        # Calculate token overlap ratio for same-language evaluation
        overlap = answer_tokens.intersection(context_tokens)
        overlap_ratio = len(overlap) / float(len(answer_tokens))

        if overlap_ratio < self.min_token_overlap_ratio:
            return False, round(overlap_ratio, 2), REFUSAL_MSG

        confidence = min(1.0, round(0.5 + overlap_ratio * 0.5, 2))
        return True, confidence, answer
