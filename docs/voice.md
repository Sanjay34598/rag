# Stage 5B: Sarvam Speech-to-Text & Voice RAG Pipeline Documentation

## 1. System Overview
The Stage 5B pipeline extends the grounded RAG architecture with voice recording and Speech-to-Text (STT) capabilities powered by Sarvam AI.

```
Browser Microphone
        ↓
MediaRecorder API (Dynamic MIME Detection, 30s Max Limit)
        ↓
FastAPI Backend (POST /api/v1/voice/query)
        ↓
Sarvam STT REST API (saaras:v3, mode=transcribe, language_code=hi-IN)
        ↓
Recognized Hindi Text Transcript
        ↓
Input & Prompt Injection Guardrails
        ↓
Hybrid Retrieval Engine (FAISS Vector + BM25 Sparse)
        ↓
Retrieval Guardrail & Context Builder
        ↓
LLM / Extracted Fallback Answer Generator
        ↓
Grounding Validator (Stopword-Filtered Token Overlap)
        ↓
Structured Response with Real Latency Breakdown
```

## 2. Sarvam STT REST Configuration
- **Model**: `SARVAM_STT_MODEL=saaras:v3` (Current recommended STT model)
- **Mode**: `SARVAM_STT_MODE=transcribe` (Preserves native spoken language without translation)
- **Language Code**: `SARVAM_LANGUAGE_CODE=hi-IN`
- **Endpoint**: `https://api.sarvam.ai/speech-to-text`
- **Form Data Fields**:
  - `model`: `"saaras:v3"`
  - `mode`: `"transcribe"`
  - `language_code`: `"hi-IN"`
  - `file`: `(filename, audio_bytes, mime_type)`

## 3. API Endpoints

### 3.1 Speech-to-Text Transcription (`POST /api/v1/stt/transcribe`)
- **Content-Type**: `multipart/form-data`
- **Body**: `file` (binary audio blob - WAV, MP3, WEBM, OGG, M4A, AAC, FLAC)
- **Response**:
```json
{
  "transcript": "कॉर्पोरेशन क्या है?",
  "language_code": "hi-IN",
  "language_probability": 0.98,
  "latency_ms": 145.2
}
```

### 3.2 Voice Query RAG Pipeline (`POST /api/v1/voice/query`)
- **Content-Type**: `multipart/form-data`
- **Body**: `file` (binary audio blob)
- **Response**:
```json
{
  "transcript": "कॉर्पोरेशन क्या है?",
  "language_code": "hi-IN",
  "language_probability": 0.98,
  "answer": "मैकडॉनल्ड कॉर्पोरेशन दुनिया के सबसे पहचानने योग्य निगमों में...",
  "grounded": true,
  "confidence": 1.0,
  "sources": [
    {
      "chunk_id": "p_1060300_5",
      "score": 0.7,
      "text": "मैकडॉनल्ड कॉर्पोरेशन दुनिया के सबसे..."
    }
  ],
  "latency": {
    "stt_ms": 145.2,
    "retrieval_ms": 34.84,
    "context_ms": 1.31,
    "llm_ms": 0.02,
    "grounding_ms": 0.32,
    "total_ms": 181.69
  }
}
```

## 4. Frontend MediaRecorder & 30s Limit
- **Dynamic Format Detection**: Evaluates `MediaRecorder.isTypeSupported(...)` across `audio/webm;codecs=opus`, `audio/webm`, `audio/ogg;codecs=opus`, `audio/mp4`, `audio/wav`.
- **Duration Limit**: Automatically stops recording at 30 seconds to comply with Sarvam REST API specs.
- **Server File Size Limit**: Enforces 10MB max upload size.

## 5. Security & Fallback Behavior
- **API Keys**: `SARVAM_API_KEY` is kept server-side.
- **Missing Key Response**: Returns `503 Service Unavailable` with `"REAL SARVAM TEST = NOT AVAILABLE"`.
- **No Hardcoded Confidence**: `language_probability` is mapped directly from Sarvam API response or set to `null` if unavailable.
