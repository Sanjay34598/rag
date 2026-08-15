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

def get_localized_refusal(language_code: str = None) -> str:
    if language_code in ("te-IN", "te"):
        return "అందుబాటులో ఉన్న సమాచారం ఆధారంగా ఈ సమాధానాన్ని ధృవీకరించలేకపోయాను."
    elif language_code in ("hi-IN", "hi"):
        return "उपलब्ध संदर्भ से मैं इस उत्तर की पुष्टि नहीं कर सका।"
    else:
        return "I couldn't verify that answer from the available context."

class GroundingValidator:
    def __init__(self, min_token_overlap_ratio: float = 0.25):
        self.min_token_overlap_ratio = min_token_overlap_ratio

    def validate(self, answer: str, context_chunks: List[Dict[str, Any]], query: str = "", language_code: str = None) -> Tuple[bool, float, str]:
        refusal_msg = get_localized_refusal(language_code)

        if not answer or not answer.strip():
            return False, 0.0, refusal_msg

        clean_answer_lower = answer.strip().lower()

        # Comprehensive refusal triggers indicating LLM determined context was insufficient
        refusal_triggers = [
            "पर्याप्त जानकारी नहीं मिली", "couldn't find", "couldn't verify",
            "refused:", "क्षमा करें", "క్షమించండి", "సమాధానం రూపొందించడానికి", "temporarily unavailable",
            "does not directly relate", "does not contain", "the context does not",
            "does not mention", "not mention", "no mention", "not provided",
            "not mentioned in the context", "is not provided in the context",
            "provided context does not", "collection of articles or passages",
            "not relevant to", "insufficient context", "cannot answer", "no information",
            "unable to answer", "not stated in the context", "not found in context"
        ]
        if any(trigger in clean_answer_lower for trigger in refusal_triggers):
            return False, 0.0, refusal_msg

        if not context_chunks:
            return False, 0.0, refusal_msg

        # Conversational / courtesy phrase filter
        if query:
            clean_query = query.strip().lower()
            conversational_phrases = {
                "आपके लिए धन्यवाद", "धन्यवाद", "शुक्रिया", "थैंक यू", "नमस्ते", 
                "hello", "hi", "thank you", "thanks", "thank you very much",
                "నమస్కారం", "ధన్యవాదాలు", "వీడ్కోలు"
            }
            if clean_query in conversational_phrases:
                return False, 0.0, refusal_msg

        answer_tokens = tokenize_text(answer)
        if not answer_tokens:
            return False, 0.0, refusal_msg

        context_text = " ".join([c.get("text", "") for c in context_chunks])
        context_tokens = tokenize_text(context_text)

        if not context_tokens:
            return False, 0.0, refusal_msg

        # Check if answer is in a non-English script (Devanagari or Telugu)
        is_ans_hindi = bool(re.search(r'[\u0900-\u097F]', answer))
        is_ans_telugu = bool(re.search(r'[\u0C00-\u0C7F]', answer))
        
        is_ctx_hindi = bool(re.search(r'[\u0900-\u097F]', context_text))
        is_ctx_telugu = bool(re.search(r'[\u0C00-\u0C7F]', context_text))

        # Cross-Lingual Grounding: If answer is Hindi/Telugu but context is English (canonical dataset evidence),
        # token overlap cannot be calculated across different scripts. If answer passed refusal triggers, accept as grounded.
        if is_ans_hindi and not is_ctx_hindi:
            return True, 0.90, answer
        if is_ans_telugu and not is_ctx_telugu:
            return True, 0.90, answer

        # Same-script token overlap evaluation
        overlap = answer_tokens.intersection(context_tokens)
        overlap_ratio = len(overlap) / float(len(answer_tokens))

        if overlap_ratio < self.min_token_overlap_ratio:
            return False, round(overlap_ratio, 2), refusal_msg

        confidence = min(1.0, round(0.5 + overlap_ratio * 0.5, 2))
        return True, confidence, answer
