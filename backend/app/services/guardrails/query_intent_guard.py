import re
from typing import Tuple

class QueryIntentGuard:
    """
    Deterministic Guard to detect non-knowledge conversational inputs 
    (greetings, thanks, farewells, casual acknowledgements) BEFORE running RAG retrieval.
    """

    CONVERSATIONAL_PHRASES = {
        # Thanks / Gratitude
        "धन्यवाद", "बहुत धन्यवाद", "आपके लिए धन्यवाद", "आपका धन्यवाद", 
        "शुक्रिया", "बहुत शुक्रिया", "धन्यवाद जी", "थैंक यू", "थैंकयू",
        "thank you", "thanks", "thanks a lot", "thank you very much",

        # Greetings
        "नमस्ते", "नमस्कार", "हैलो", "हेलो", "hello", "hi", "hey", "good morning", "good evening",

        # Farewells
        "अलविदा", "फिर मिलेंगे", "goodbye", "bye", "bye bye",

        # Acknowledgements / Casual
        "ठीक है", "ठीक", "अच्छा", "हाँ", "जी", "okay", "ok"
    }

    # Indicators that a query is asking for DEFINITION, MEANING, or EXPLANATION (Knowledge Query)
    KNOWLEDGE_QUERY_INDICATORS = {
        "क्या", "परिभाषा", "अर्थ", "विवरण", "कारण", "क्यों", "कैसे", "कौन", "किसे", "कहाँ", "कब",
        "what", "define", "definition", "meaning", "explain", "why", "how", "who", "where", "when"
    }

    def evaluate(self, query: str) -> Tuple[bool, str]:
        """
        Returns (is_conversational: bool, safe_response: str)
        """
        if not query or not query.strip():
            return False, ""

        clean_query = query.strip().lower()
        clean_query_no_punct = re.sub(r'[\.?!।]+$', '', clean_query).strip()

        # Check if query contains explicit knowledge query indicators (e.g. "क्या", "अर्थ", "परिभाषा")
        tokens = re.findall(r'[\w\u0900-\u097F]+', clean_query_no_punct)
        token_set = set(tokens)

        # If the query contains knowledge indicators, it IS a knowledge query
        if token_set.intersection(self.KNOWLEDGE_QUERY_INDICATORS):
            return False, ""

        # Check exact or phrase match against conversational phrases
        if clean_query_no_punct in self.CONVERSATIONAL_PHRASES:
            return True, self._get_conversational_response(clean_query_no_punct)

        for phrase in self.CONVERSATIONAL_PHRASES:
            if clean_query_no_punct == phrase:
                return True, self._get_conversational_response(phrase)

        return False, ""

    def _get_conversational_response(self, phrase: str) -> str:
        if any(w in phrase for w in ["thank", "thanks", "धन्यवाद", "शुक्रिया"]):
            return "You're welcome! / आपका स्वागत है।"
        elif any(w in phrase for w in ["hello", "hi", "hey", "नमस्ते", "नमस्कार", "हैलो"]):
            return "Hello! How can I help you today? / नमस्ते! मैं आपकी क्या सहायता कर सकता हूँ?"
        elif any(w in phrase for w in ["bye", "goodbye", "अलविदा", "फिर मिलेंगे"]):
            return "Goodbye! Have a great day! / अलविदा! आपका दिन शुभ हो।"
        else:
            return "Got it! Please ask any knowledge question you have. / ठीक है! कृपया अपना ज्ञान प्रश्न पूछें।"
