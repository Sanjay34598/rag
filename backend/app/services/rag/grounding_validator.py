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
    text = re.sub(r'[\u0964\u0965]', ' ', text)
    raw_tokens = [t.strip('।,.:;!?()[]{}""\'\'`') for t in text.split()]
    return {t for t in raw_tokens if len(t) >= 2 and t not in HINDI_ENGLISH_TELUGU_STOPWORDS}

def get_localized_refusal(language_code: str = None) -> str:
    if language_code in ("te-IN", "te"):
        return "అందుబాటులో ఉన్న సమాచారం ఆధారంగా ఈ సమాధానాన్ని ధృవీకరించలేకపోయాను."
    elif language_code in ("hi-IN", "hi"):
        return "उपलब्ध संदर्भ से मैं इस उत्तर की पुष्टि नहीं कर सका।"
    else:
        return "I couldn't verify that answer from the available context."

def validate_language_output(answer: str, language_code: str = None) -> bool:
    """Helper to verify that the answer contains the expected target script characters."""
    if not answer or not answer.strip():
        return False
    if language_code in ("hi-IN", "hi"):
        return bool(re.search(r'[\u0900-\u097F]', answer))
    elif language_code in ("te-IN", "te"):
        return bool(re.search(r'[\u0C00-\u0C7F]', answer))
    return True

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
            "does not mention", "not mentioned in the context", "is not provided in the context",
            "provided context does not", "collection of articles or passages",
            "not relevant to", "insufficient context", "cannot answer", "no information",
            "unable to answer", "not stated in the context", "not found in context",
            "पुष्टि नहीं मिली", "धृवीకరించలేకపోయాను"
        ]
        if any(trigger in clean_answer_lower for trigger in refusal_triggers):
            return False, 0.0, refusal_msg

        if not context_chunks:
            return False, 0.0, refusal_msg

        # Intent alignment validation: verify context chunks support query intent
        if query:
            from app.services.retrieval.vitamin_expansion import extract_query_intent, does_chunk_support_intent
            q_intent = extract_query_intent(query)
            valid_chunks = [c for c in context_chunks if does_chunk_support_intent(c, q_intent)[0]]
            if not valid_chunks:
                print(f"[GROUNDING VALIDATOR REJECTION] No context chunk supports intent '{q_intent}' for query '{query}'")
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

        # Check if answer is in a non-English script (Devanagari or Telugu)
        is_ans_hindi = bool(re.search(r'[\u0900-\u097F]', answer))
        is_ans_telugu = bool(re.search(r'[\u0C00-\u0C7F]', answer))

        # Include translated text fields from context chunks for multilingual token validation
        context_parts = []
        for c in context_chunks:
            context_parts.append(c.get("text", ""))
            if is_ans_hindi or language_code in ("hi-IN", "hi"):
                if c.get("translated_text_hi"):
                    context_parts.append(c.get("translated_text_hi"))
            if is_ans_telugu or language_code in ("te-IN", "te"):
                if c.get("translated_text_te"):
                    context_parts.append(c.get("translated_text_te"))

        context_text = " ".join(context_parts)
        context_tokens = tokenize_text(context_text)

        if not context_tokens:
            return False, 0.0, refusal_msg

        is_ctx_hindi = bool(re.search(r'[\u0900-\u097F]', context_text))
        is_ctx_telugu = bool(re.search(r'[\u0C00-\u0C7F]', context_text))

        # Multilingual terminology entity mapping for cross-lingual token overlap evaluation
        ENTITY_MAP = {
            # Corporation, business & legal terminology
            "निगम": "corporation", "कॉर्पोरेशन": "corporation",
            "कल्याण": "welfare", "संस्था": "entity", "संगठन": "organization",
            "कानून": "legal", "कानूनी": "legal", "स्थापित": "established",
            "व्यक्ति": "person", "व्यक्तियों": "persons", "संघ": "association",
            "व्यापार": "business", "कंपनी": "company",
            "కార్పొరేషన్": "corporation", "సంస్థ": "corporation", "చట్టబద్ధమైన": "legal",
            "చట్టం": "law", "చట్టపరమైన": "legal", "వ్యక్తుల": "persons", "వ్యాపారం": "business",
            
            # Vitamin & chemical terminology
            "विटामिन": "vitamin", "विटामिनों": "vitamin",
            "राइबोफ्लेविन": "riboflavin", "थायमिन": "thiamine",
            "नायसिन": "niacin", "फॉलिक": "folic", "कोबालामिन": "cobalamin",
            "पाइरिडोक्सिन": "pyridoxine", "बायोटिन": "biotin",
            "విటమిన్": "vitamin", "విటమిన్లు": "vitamin",
            "రిబోఫ్లావిన్": "riboflavin", "థయామిన్": "thiamine",
            "నియాసిన్": "niacin", "ఫొలిక్": "folic", "కోబాలమిన్": "cobalamin",
            "పైరిడాక్సిన్": "pyridoxine", "బయోటిన్": "biotin"
        }

        context_text_lower = context_text.lower()
        overlap = set()
        for t in answer_tokens:
            if t in context_tokens or (t in ENTITY_MAP and ENTITY_MAP[t] in context_tokens):
                overlap.add(t)
            elif len(t) >= 5:
                stem = t[:4]
                if any(ct.startswith(stem) for ct in context_tokens):
                    overlap.add(t)

        overlap_ratio = len(overlap) / float(len(answer_tokens)) if answer_tokens else 0.0

        if overlap_ratio < self.min_token_overlap_ratio:
            return False, round(overlap_ratio, 2), refusal_msg

        confidence = min(1.0, round(0.5 + overlap_ratio * 0.5, 2))
        return True, confidence, answer
