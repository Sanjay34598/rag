import time
import requests
from typing import Tuple, Dict, Any
from app.core.config import settings

class SarvamSTTService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SarvamSTTService, cls).__new__(cls)
        return cls._instance

    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        mime_type: str = "audio/wav"
    ) -> Tuple[bool, Dict[str, Any], float]:
        """
        Transcribe audio using Sarvam AI Speech-to-Text API.
        Returns: (success: bool, result_dict: dict, latency_ms: float)
        """
        start_time = time.perf_counter()

        if not settings.SARVAM_API_KEY:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return False, {
                "error": "Sarvam STT API key not configured (SARVAM_API_KEY is missing)",
                "code": "MISSING_API_KEY"
            }, round(latency_ms, 2)

        if not audio_bytes or len(audio_bytes) == 0:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return False, {
                "error": "Audio content is empty",
                "code": "EMPTY_AUDIO"
            }, round(latency_ms, 2)

        headers = {
            "api-subscription-key": settings.SARVAM_API_KEY
        }

        files = {
            "file": (filename, audio_bytes, mime_type)
        }

        data = {
            "model": settings.SARVAM_STT_MODEL,
            "language_code": "hi-IN"
        }

        try:
            response = requests.post(
                settings.SARVAM_STT_URL,
                headers=headers,
                files=files,
                data=data,
                timeout=settings.SARVAM_TIMEOUT
            )
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            if response.status_code != 200:
                return False, {
                    "error": f"Sarvam STT API failed with HTTP {response.status_code}",
                    "details": response.text[:200],
                    "code": "API_ERROR"
                }, round(latency_ms, 2)

            res_data = response.json()
            transcript = res_data.get("transcript", "")
            language_code = res_data.get("language_code", "hi-IN")

            if not transcript or not transcript.strip():
                return False, {
                    "error": "Empty transcription returned by Sarvam STT",
                    "code": "EMPTY_TRANSCRIPT"
                }, round(latency_ms, 2)

            result = {
                "transcript": transcript.strip(),
                "language": language_code,
                "confidence": 1.0,
                "latency_ms": round(latency_ms, 2)
            }
            return True, result, round(latency_ms, 2)

        except requests.exceptions.Timeout:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return False, {
                "error": "Sarvam STT API call timed out",
                "code": "TIMEOUT"
            }, round(latency_ms, 2)

        except requests.exceptions.RequestException as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return False, {
                "error": f"Sarvam STT API connection failed: {str(e)}",
                "code": "CONNECTION_ERROR"
            }, round(latency_ms, 2)

def get_stt_service() -> SarvamSTTService:
    return SarvamSTTService()
