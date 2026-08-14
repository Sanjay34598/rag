import os
import json
import time
import requests
from typing import Dict, Any, List
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
        self.model = settings.LLM_MODEL
        self.api_key = settings.LLM_API_KEY
        self.max_retries = settings.LLM_MAX_RETRIES
        self.timeout = settings.LLM_TIMEOUT
        self._initialized = True
        print(f"[AnswerGenerator] Initialized LLM generator (Mode: {self.mode}, Provider: {self.provider}, Model: {self.model}, API Key Set: {bool(self.api_key)})")

    def _call_gemini_api(self, prompt: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 500}
        }
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("No candidates returned from Gemini API")
        
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            raise RuntimeError("Empty content returned from Gemini API")
            
        return parts[0].get("text", "")

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

    def _offline_fallback_generate(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extracted grounded answer fallback when offline mode is requested or no API key is available.
        """
        REFUSAL_MSG = "मुझे इस प्रश्न का विश्वसनीय उत्तर देने के लिए पर्याप्त जानकारी नहीं मिली।"

        if not context_chunks:
            return {
                "answer": REFUSAL_MSG,
                "grounded": False,
                "confidence": 0.0,
                "llm_mode": "fallback"
            }
            
        top_chunk = context_chunks[0]
        score = top_chunk.get("score", 0.0)
        dense_s = top_chunk.get("dense_score", 0.0)
        bm25_s = top_chunk.get("bm25_score", 0.0)

        if score < settings.MIN_RETRIEVAL_SCORE or (dense_s < 0.22 and bm25_s < 3.0):
            return {
                "answer": REFUSAL_MSG,
                "grounded": False,
                "confidence": 0.0,
                "llm_mode": "fallback"
            }
        
        text = top_chunk.get("text", "").strip()
        
        return {
            "answer": text,
            "grounded": True,
            "confidence": round(min(1.0, float(score)), 2),
            "llm_mode": "fallback"
        }

    def generate(self, query: str, prompt: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.mode == "fallback" or not self.api_key:
            return self._offline_fallback_generate(query, context_chunks)

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                if "openai" in self.provider:
                    raw_output = self._call_openai_api(prompt)
                else:
                    raw_output = self._call_gemini_api(prompt)

                parsed = self._parse_llm_json(raw_output)
                parsed["llm_mode"] = "real"
                return parsed
            except Exception as e:
                last_error = e
                print(f"[AnswerGenerator] Real LLM API call attempt {attempt+1} failed: {e}")
                time.sleep(0.5 * (attempt + 1))

        print(f"[AnswerGenerator] All LLM API retries failed ({last_error}). Falling back to extracted context answer.")
        res = self._offline_fallback_generate(query, context_chunks)
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
