import asyncio
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.v1.api import api_router
from app.services.rag.rag_service import get_rag_service

# Safe UTF-8 initialization for stdout/stderr on Windows/CP1252
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

print("[BOOT] main.py imported")

async def initialize_rag_background():
    try:
        service = get_rag_service()
        await asyncio.to_thread(service.initialize, load_indexes=True)
    except Exception as e:
        print(f"[RAG INIT FAILED] {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[STARTUP] lifespan entered")
    print("[STARTUP] scheduling background RAG initialization")
    asyncio.create_task(initialize_rag_background())
    print("[STARTUP] yielding control to Uvicorn")
    print("[STARTUP] HTTP application ready")
    yield
    print("[SHUTDOWN] Application shutting down")

print("[BOOT] FastAPI app creation started")
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)
print("[BOOT] FastAPI app created")

import os

raw_cors_origins = os.getenv("CORS_ORIGINS") or os.getenv("ALLOWED_ORIGINS") or getattr(settings, "CORS_ORIGINS", "")
cors_origins = [o.strip() for o in raw_cors_origins.split(",") if o.strip()]
if not cors_origins:
    cors_origins = [
        "https://voice-b0064qrq6-sanjays-projects-f2a71297.vercel.app",
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True if "*" not in cors_origins else False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "version": settings.VERSION,
        "docs": "/docs"
    }

@app.get("/health")
def health():
    service = get_rag_service()
    return {
        "status": "ok",
        "version": settings.VERSION,
        "rag_ready": getattr(service, "is_ready", False)
    }

@app.get("/cors-test")
def cors_test():
    return {
        "status": "ok",
        "message": "CORS is working"
    }

