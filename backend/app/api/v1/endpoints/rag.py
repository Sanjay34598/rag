from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from app.services.rag.rag_service import get_rag_service

router = APIRouter()

class RAGQueryRequest(BaseModel):
    query: str = Field(..., description="User search query", example="kya vitamin b ka atyadhik sevan hanikarak hai?")
    language_code: Optional[str] = Field(None, description="Optional target language: hi-IN, en-IN, te-IN, unknown")
    language: Optional[str] = Field(None, description="Alternative field for language code")

class RAGSource(BaseModel):
    chunk_id: str
    language: Optional[str] = None
    score: float
    text: str

class RAGLatency(BaseModel):
    retrieval_ms: float
    context_ms: float
    llm_ms: float
    grounding_ms: float
    total_ms: float

class RAGQueryResponse(BaseModel):
    query: str
    answer: str
    grounded: bool
    confidence: float
    sources: List[RAGSource]
    latency: RAGLatency
    language_code: Optional[str] = None

@router.post("/query", response_model=RAGQueryResponse, summary="Perform Grounded RAG Query")
def rag_query_endpoint(request: RAGQueryRequest):
    service = get_rag_service()
    if not getattr(service, "is_ready", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system is still initializing. Please try again in a few seconds."
        )

    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty or whitespace-only."
        )

    target_lang = request.language_code or request.language
    try:
        service = get_rag_service()
        response_data = service.answer(query=request.query.strip(), language_code=target_lang)
        return response_data
    except Exception as e:
        safe_err = str(e).encode("utf-8", errors="replace").decode("utf-8")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG execution error: {safe_err}"
        )

