import os
import re
import json
import time
import requests
from typing import Dict, Any, List
from groq import Groq
from app.core.config import settings

class AnswerGenerator:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AnswerGenerator, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.mode = settings.LLM_MODE.lower()
        self.provider = settings.LLM_PROVIDER.lower()
        self.model = settings.LLM_MODEL if settings.LLM_MODEL and "llama" in settings.LLM_MODEL else "llama-3.1-8b-instant"
        self.api_key = settings.GROQ_API_KEY or settings.LLM_API_KEY
        self.max_retries = settings.LLM_MAX_RETRIES
        self.timeout = settings.LLM_TIMEOUT
        
        self.groq_client = None
        if self.api_key:
            try:
                self.groq_client = Groq(api_key=self.api_key)
            except Exception as e:
                print(f"[AnswerGenerator] Failed to initialize Groq client: {self._sanitize_string(str(e))}")

        self._initialized = True
        print(f"[AnswerGenerator] Initialized LLM generator (Mode: {self.mode}, Provider: {self.provider}, Model: {self.model}, API Key Set: {bool(self.api_key)})")

    def _sanitize_string(self, text: str) -> str:
        if not text:
            return text
        text = re.sub(r'key=[^&\s"\']+', 'key=[REDACTED]', text)
        text = re.sub(r'gsk_[A-Za-z0-9_]+', 'gsk_[REDACTED]', text)
        text = re.sub(r'Bearer\s+[A-Za-z0-9\-\._~\+\/]+=*', 'Bearer [REDACTED]', text)
        return text

    def _call_groq_api(self, prompt: str) -> str:
        if not self.groq_client:
            self.groq_client = Groq(api_key=self.api_key)

        chat_completion = self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=0.2,
            response_format={"type": "json_object"},
            timeout=self.timeout
        )
        return chat_completion.choices[0].message.content or ""

    def _call_openai_api(self, prompt: str) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model if "gpt" in self.model else "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "response_format": {"type": "json_object"}
        }
        response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _offline_fallback_generate(self, query: str, context_chunks: List[Dict[str, Any]], language_code: str = None) -> Dict[str, Any]:
        """
        Extracted localized fallback message when offline mode, 429 rate limit, or API failure occurs.
        NEVER returns raw Hindi context text as a final answer for Telugu or English queries.
        """
        if self.mode == "fallback" and context_chunks:
            top_chunk = context_chunks[0]
            score = float(top_chunk.get("score", 0.0))
            if score >= settings.MIN_RETRIEVAL_SCORE:
                if (language_code or "hi-IN") == "hi-IN":
                    return {
                        "answer": top_chunk.get("text", "").strip(),
                        "grounded": True,
                        "confidence": round(min(1.0, score), 2),
                        "llm_mode": "fallback"
                    }

        if language_code == "te-IN":
            refusal_msg = "అందుబాటులో ఉన్న సమాచారం ఆధారంగా ఆ సమాధానాన్ని ధృవీకరించలేకపోయాను."
        elif language_code == "en-IN":
            refusal_msg = "I couldn't verify that answer from the available context."
        else:
            refusal_msg = "मुझे उपलब्ध संदर्भ से उस उत्तर की पुष्टि नहीं मिली।"

        return {
            "answer": refusal_msg,
            "grounded": False,
            "confidence": 0.0,
            "llm_mode": "fallback_refusal"
        }

    def generate(self, query: str, prompt: str, context_chunks: List[Dict[str, Any]], language_code: str = None) -> Dict[str, Any]:
        if self.mode == "fallback" or not self.api_key:
            return self._offline_fallback_generate(query, context_chunks, language_code=language_code)

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                if "openai" in self.provider:
                    raw_output = self._call_openai_api(prompt)
                else:
                    raw_output = self._call_groq_api(prompt)

                parsed = self._parse_llm_json(raw_output)
                parsed["llm_mode"] = "real"
                return parsed
            except Exception as e:
                last_error = e
                sanitized_err = self._sanitize_string(str(e))
                try:
                    print(f"[LLM DEBUG] Real Groq LLM API call attempt {attempt+1} failed: {sanitized_err}")
                except Exception:
                    pass

                # Fail fast on 429, 401, 403, 400 errors without sequential retries
                err_code = getattr(e, 'status_code', None) or getattr(getattr(e, 'response', None), 'status_code', None)
                if err_code in (429, 401, 403, 400) or "429" in str(e) or "401" in str(e):
                    try:
                        print(f"[LLM DEBUG] Groq API error (status: {err_code}). Failing fast without retries.")
                    except Exception:
                        pass
                    break

                if attempt < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))

        try:
            print(f"[LLM DEBUG] Groq LLM API call failed ({self._sanitize_string(str(last_error))}). Invoking localized fallback refusal path.")
        except Exception:
            pass
        res = self._offline_fallback_generate(query, context_chunks, language_code=language_code)
        res["llm_mode"] = "fallback_after_error"
        return res

    def _parse_llm_json(self, raw_text: str) -> Dict[str, Any]:
        cleaned = raw_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
            return {
                "answer": str(parsed.get("answer", cleaned)),
                "grounded": bool(parsed.get("grounded", True)),
                "confidence": float(parsed.get("confidence", 0.85))
            }
        except Exception:
            return {
                "answer": cleaned,
                "grounded": True,
                "confidence": 0.80
            }

def get_answer_generator() -> AnswerGenerator:
    return AnswerGenerator()
