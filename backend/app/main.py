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

def _sync_initialize_rag():
    try:
        service = get_rag_service()
        service.initialize(load_indexes=True)
    except Exception as e:
        print(f"[{settings.PROJECT_NAME}] Warning during background RAG model loading: {e}")

async def _bg_initialize_rag():
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _sync_initialize_rag)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[STARTUP] FastAPI application starting")
    print(f"[{settings.PROJECT_NAME}] Pre-loading RAG & retrieval models asynchronously...")
    init_task = asyncio.create_task(_bg_initialize_rag())
    print("[STARTUP] HTTP application ready")
    yield
    print(f"[{settings.PROJECT_NAME}] Shutting down...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

cors_origins = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()]
if not cors_origins:
    cors_origins = ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173"]

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
