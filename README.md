# Voice-Enabled RAG System (HH Goa 2026 Task 2)

A fast, reliable Voice-Enabled Retrieval-Augmented Generation (RAG) system built with FastAPI, React + Vite, Sarvam Speech-to-Text, FAISS + BM25 Hybrid Retrieval, and LLM answer generation.

## Project Architecture

`Voice Input` → `Sarvam STT` → `Query Processing` → `Hybrid Retrieval (FAISS + BM25)` → `Reranker` → `Guardrails` → `LLM` → `Grounded Answer` → `React UI`

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
Copy `.env.example` to `.env` and update API keys:
```bash
cp .env.example .env
```
