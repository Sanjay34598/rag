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
        "నమస్కారం", "ధన్యవాదాలు", "ధన్యవాదములు", "చాలా ధన్యవాదాలు", "థాంక్యూ",

        # Greetings
        "नमस्ते", "नमस्कार", "हैलो", "हेलो", "hello", "hi", "hey", "good morning", "good evening",
        "నమస్తే", "హలో", "గుడ్ మార్నింగ్",

        # Farewells
        "अलविदा", "फिर मिलेंगे", "goodbye", "bye", "bye bye",
        "వీడ్కోలు", "మళ్ళీ కలుద్దాం",

        # Acknowledgements / Casual
        "ठीक है", "ठीक", "अच्छा", "हाँ", "जी", "okay", "ok"
    }

    # Indicators that a query is asking for DEFINITION, MEANING, or EXPLANATION (Knowledge Query)
    KNOWLEDGE_QUERY_INDICATORS = {
        # Hindi
        "क्या", "परिभाषा", "अर्थ", "विवरण", "कारण", "क्यों", "कैसे", "कौन", "किसे", "कहाँ", "कब",
        # English
        "what", "define", "definition", "meaning", "explain", "why", "how", "who", "where", "when", "does", "is",
        # Telugu
        "ఏమిటి", "ఏంటి", "అర్థం", "నిర్వచనం", "వివరించండి", "ఎలా", "ఎందుకు", "ఎవరు", "ఎక్కడ", "ఎప్పుడు", "గురించి"
    }

    def evaluate(self, query: str, language_code: str = None) -> Tuple[bool, str]:
        """
        Returns (is_conversational: bool, safe_response: str)
        """
        if not query or not query.strip():
            return False, ""

        clean_query = query.strip().lower()
        clean_query_no_punct = re.sub(r'[\.?!।]+$', '', clean_query).strip()

        # Check if query contains explicit knowledge query indicators
        # Supports ASCII, Devanagari (\u0900-\u097F), and Telugu (\u0C00-\u0C7F)
        tokens = re.findall(r'[\w\u0900-\u097F\u0C00-\u0C7F]+', clean_query_no_punct)
        token_set = set(tokens)

        # If the query contains knowledge indicators, it IS a knowledge query
        if token_set.intersection(self.KNOWLEDGE_QUERY_INDICATORS):
            return False, ""

        # Check exact or phrase match against conversational phrases
        if clean_query_no_punct in self.CONVERSATIONAL_PHRASES:
            return True, self._get_conversational_response(clean_query_no_punct, language_code)

        for phrase in self.CONVERSATIONAL_PHRASES:
            if clean_query_no_punct == phrase:
                return True, self._get_conversational_response(phrase, language_code)

        return False, ""

    def _get_conversational_response(self, phrase: str, language_code: str = None) -> str:
        lang = language_code or "hi-IN"
        if any(w in phrase for w in ["thank", "thanks", "धन्यवाद", "शुक्रिया", "ధన్యవాదాలు", "ధన్యవాదములు", "థాంక్యూ"]):
            if lang == "te-IN":
                return "మీకు స్వాగతం!"
            elif lang == "en-IN":
                return "You're welcome!"
            else:
                return "आपका स्वागत है!"
        elif any(w in phrase for w in ["hello", "hi", "hey", "नमस्ते", "नमस्कार", "हैलो", "నమస్కారం", "నమస్తే", "హలో"]):
            if lang == "te-IN":
                return "నమస్కారం! నేను మీకు ఎలా సహాయపడగలను?"
            elif lang == "en-IN":
                return "Hello! How can I help you today?"
            else:
                return "नमस्ते! मैं आपकी क्या सहायता कर सकता हूँ?"
        elif any(w in phrase for w in ["bye", "goodbye", "अलविदा", "फिर मिलेंगे", "వీడ్కోలు"]):
            if lang == "te-IN":
                return "వీడ్కోలు! మీ రోజు శుభప్రదంగా ఉండుగాక."
            elif lang == "en-IN":
                return "Goodbye! Have a great day!"
            else:
                return "अलविदा! आपका दिन शुभ हो।"
        else:
            if lang == "te-IN":
                return "సరే! దయచేసి మీ రంగానికి సంబంధించిన ప్రశ్నను అడగండి."
            elif lang == "en-IN":
                return "Got it! Please ask any knowledge question you have."
            else:
                return "ठीक है! कृपया अपना ज्ञान संबंधी प्रश्न पूछें।"
