import time
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, status
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from app.services.stt.sarvam_stt import get_stt_service
from app.services.rag.rag_service import get_rag_service
from app.api.v1.endpoints.stt import validate_audio_file

router = APIRouter()

class VoiceLatencyDetail(BaseModel):
    stt_ms: float
    retrieval_ms: float
    context_ms: float
    llm_ms: float
    grounding_ms: float
    total_ms: float

class VoiceQueryResponse(BaseModel):
    transcript: str
    language_code: Optional[str] = "hi-IN"
    language_probability: Optional[float] = None
    answer: str
    grounded: bool
    confidence: float
    sources: List[Dict[str, Any]]
    latency: VoiceLatencyDetail

@router.post("/query", response_model=VoiceQueryResponse)
async def voice_query(
    file: UploadFile = File(...),
    language_code: Optional[str] = Form(None),
    language: Optional[str] = Form(None)
):
    """
    Voice-enabled RAG Pipeline:
    Audio -> Sarvam STT -> Transcript -> Hybrid RAG -> Answer + Grounding -> Response
    """
    rag_service = get_rag_service()
    if not getattr(rag_service, "is_ready", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system is still initializing. Please try again in a few seconds."
        )

    total_start = time.perf_counter()
    content_bytes = await file.read()
    validate_audio_file(file, content_bytes)

    stt_service = get_stt_service()
    filename = file.filename or "audio.wav"
    mime_type = file.content_type or "audio/wav"

    requested_lang = language_code or language

    success, stt_res, stt_ms = stt_service.transcribe(
        audio_bytes=content_bytes,
        filename=filename,
        mime_type=mime_type,
        language_code=requested_lang
    )

    if not success:
        error_msg = stt_res.get("error", "Speech-to-Text transcription failed.")
        code = stt_res.get("code", "STT_ERROR")
        upstream_status = stt_res.get("status_code", 500)

        if code == "MISSING_API_KEY":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Voice Query unavailable: {error_msg}"
            )
        elif code in ("EMPTY_TRANSCRIPT", "EMPTY_AUDIO"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_msg
            )
        elif code == "TIMEOUT":
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=error_msg
            )
        elif code == "API_ERROR":
            if 400 <= upstream_status < 500:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=error_msg
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=error_msg
                )
        elif code == "CONNECTION_ERROR":
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=error_msg
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_msg
            )

    transcript = stt_res["transcript"]
    # User selected language overrides STT detected language for RAG output generation
    effective_lang = requested_lang or stt_res.get("language_code") or "hi-IN"

    rag_service = get_rag_service()
    rag_res = rag_service.answer(transcript, language_code=effective_lang)

    total_ms = (time.perf_counter() - total_start) * 1000.0

    latency_breakdown = VoiceLatencyDetail(
        stt_ms=round(stt_ms, 2),
        retrieval_ms=round(rag_res["latency"]["retrieval_ms"], 2),
        context_ms=round(rag_res["latency"]["context_ms"], 2),
        llm_ms=round(rag_res["latency"]["llm_ms"], 2),
        grounding_ms=round(rag_res["latency"]["grounding_ms"], 2),
        total_ms=round(total_ms, 2)
    )

    return VoiceQueryResponse(
        transcript=transcript,
        language_code=effective_lang or "hi-IN",
        language_probability=stt_res.get("language_probability"),
        answer=rag_res["answer"],
        grounded=rag_res["grounded"],
        confidence=rag_res["confidence"],
        sources=rag_res["sources"],
        latency=latency_breakdown
    )
