from fastapi import APIRouter, File, UploadFile, HTTPException, status
from pydantic import BaseModel
from typing import Optional
from app.services.stt.sarvam_stt import get_stt_service

router = APIRouter()

MAX_AUDIO_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit
ALLOWED_MIME_TYPES = {
    "audio/wav", "audio/x-wav", "audio/mp3", "audio/mpeg", 
    "audio/webm", "audio/ogg", "audio/m4a", "audio/aac", "audio/flac"
}
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".webm", ".ogg", ".m4a", ".aac", ".flac"}

class STTTranscribeResponse(BaseModel):
    transcript: str
    language_code: Optional[str] = "hi-IN"
    language_probability: Optional[float] = None
    latency_ms: float

def validate_audio_file(file: UploadFile, content_bytes: bytes):
    if not content_bytes or len(content_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded audio file is empty."
        )

    if len(content_bytes) > MAX_AUDIO_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file size exceeds limit of {MAX_AUDIO_SIZE_BYTES // (1024 * 1024)}MB."
        )

    filename = file.filename or "audio.wav"
    ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
    mime_type = (file.content_type or "").lower().split(";")[0].strip()

    if mime_type and mime_type not in ALLOWED_MIME_TYPES and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported audio format '{file.content_type}'. Supported formats: WAV, MP3, WEBM, OGG, M4A, AAC, FLAC."
        )

@router.post("/transcribe", response_model=STTTranscribeResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe audio file using Sarvam Speech-to-Text API.
    """
    content_bytes = await file.read()
    validate_audio_file(file, content_bytes)

    stt_service = get_stt_service()
    filename = file.filename or "audio.wav"
    mime_type = file.content_type or "audio/wav"

    success, result, latency_ms = stt_service.transcribe(content_bytes, filename, mime_type)

    if not success:
        error_msg = result.get("error", "Transcription failed.")
        code = result.get("code", "STT_ERROR")
        
        if code == "MISSING_API_KEY":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=error_msg
            )
        elif code == "EMPTY_TRANSCRIPT":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_msg
            )
        elif code == "TIMEOUT":
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=error_msg
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_msg
            )

    return STTTranscribeResponse(
        transcript=result["transcript"],
        language_code=result.get("language_code", "hi-IN"),
        language_probability=result.get("language_probability"),
        latency_ms=latency_ms
    )
