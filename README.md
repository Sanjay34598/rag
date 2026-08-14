# Voice-Enabled RAG System (HH Goa 2026 Task 2)

A fast, low-latency Voice-Enabled Retrieval-Augmented Generation (RAG) system built with FastAPI, React + Vite, Sarvam Speech-to-Text, FAISS + BM25 Hybrid Retrieval, and LLM answer generation with grounding guardrails.

## Pipeline Architecture

```
Browser Microphone
        ↓
MediaRecorder API
        ↓
FastAPI Backend (POST /api/v1/voice/query)
        ↓
Sarvam Speech-to-Text API (saarika:v1 / saaras:v1)
        ↓
Recognized Hindi Text Transcript
        ↓
Input & Prompt Injection Guardrails
        ↓
Hybrid Retrieval Engine (FAISS Vector + BM25 Sparse, Weight: 0.7/0.3)
        ↓
Retrieval Guardrail & Context Builder
        ↓
LLM / Extracted Offline Fallback Answer Generator
        ↓
Grounding Validator (Stopword-Filtered Token Overlap)
        ↓
Output Guardrail
        ↓
Structured Response with Latency Breakdown (STT, Retrieval, Context, LLM, Grounding, Total)
```

## Documentation & Benchmarks
- [`docs/retrieval.md`](docs/retrieval.md): FAISS + BM25 Hybrid Indexing & Precision Weight Selection.
- [`docs/performance_audit.md`](docs/performance_audit.md): Comprehensive Latency Distribution Audit (Retrieval P100 = 179.75ms).
- [`docs/rag.md`](docs/rag.md): Stage 4 & 5A RAG Pipeline Architecture & Grounding Validator.
- [`docs/voice.md`](docs/voice.md): Stage 5B Voice RAG, Sarvam STT, and Audio Upload Specs.

## Quick Start

### 1. Backend Setup
```bash
cd backend
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### 3. Environment Variables
Copy `.env.example` to `.env` and configure keys:
```bash
cp .env.example .env
```
Key settings:
- `SARVAM_API_KEY`: Your Sarvam AI STT subscription key.
- `LLM_API_KEY`: Gemini / OpenAI / Groq API key (defaults to fallback if empty).

### 4. Running Test Suites
```bash
pytest backend/tests/ -v
```
