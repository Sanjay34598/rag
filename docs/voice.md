# Stage 5B: Sarvam Speech-to-Text & Voice RAG Pipeline Documentation

## 1. System Overview
The Stage 5B pipeline extends the grounded RAG architecture with voice recording and Speech-to-Text (STT) capabilities powered by Sarvam AI.

```
Browser Microphone
        ↓
MediaRecorder API (Local Blob)
        ↓
FastAPI Backend (POST /api/v1/voice/query)
        ↓
Sarvam STT Service (saarika:v1 / saaras:v1)
        ↓
Recognized Hindi Transcript
        ↓
Input Guardrail & Hybrid Retrieval (FAISS + BM25)
        ↓
Retrieval Guardrail & Context Builder
        ↓
LLM / Extracted Fallback Answer Generator
        ↓
Grounding Validator (Stopword-filtered token overlap)
        ↓
Structured Response with Latency Breakdown
```

## 2. API Endpoints

### 2.1 Speech-to-Text Transcription (`POST /api/v1/stt/transcribe`)
- **Content-Type**: `multipart/form-data`
- **Body**: `file` (binary audio blob - WAV, MP3, WEBM, OGG, M4A, AAC, FLAC)
- **Response**:
```json
{
  "transcript": "कॉर्पोरेशन क्या है?",
  "language": "hi-IN",
  "confidence": 1.0,
  "latency_ms": 145.2
}
```

### 2.2 Voice Query RAG Pipeline (`POST /api/v1/voice/query`)
- **Content-Type**: `multipart/form-data`
- **Body**: `file` (binary audio blob)
- **Response**:
```json
{
  "transcript": "कॉर्पोरेशन क्या है?",
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

## 3. Configuration & Security
- **API Key Configuration**: `SARVAM_API_KEY` set via environment variable or `.env`.
- **Missing API Key Behavior**: If `SARVAM_API_KEY` is not set, API returns `503 Service Unavailable` with structured detail `"REAL SARVAM TEST = NOT AVAILABLE"`. The application continues running without crashing.
- **Secrets Protection**: API keys remain strictly server-side and are never logged or exposed in HTTP responses.
- **Audio Cleanup**: Audio streams are processed in-memory (`UploadFile.read()`) without permanent disk storage.

## 4. Hindi Language Processing
- Audio input is transcribed natively in Hindi Devanagari script.
- The raw transcript is passed directly to the hybrid retrieval engine without translation.
- Hindi Devanagari regex tokenization and custom stopword filtering (`का`, `के`, `की`, `है`, `और`, etc.) preserve accurate grounding verification.

## 5. Latency Instrumentation
The pipeline measures and breaks down processing time into discrete stages:
1. `stt_ms`: Time taken for Sarvam STT API call.
2. `retrieval_ms`: Dense FAISS vector + BM25 hybrid search.
3. `context_ms`: Context string formatting and prompt assembly.
4. `llm_ms`: Answer generation (Cloud API or Fallback).
5. `grounding_ms`: Grounding validation token overlap check.
6. `total_ms`: End-to-end server request duration.

## 6. Known Limitations & Scope Bounds
- **Single-shot Audio**: User records audio locally and sends a single chunk on stop recording (continuous audio streaming / VAD is intentionally out of scope).
- **Format Support**: Standard web audio containers (`audio/webm`, `audio/wav`, `audio/mp3`, `audio/ogg`).
