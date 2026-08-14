from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from app.services.rag.rag_service import get_rag_service

router = APIRouter()

class RAGQueryRequest(BaseModel):
    query: str = Field(..., description="User search query", example="kya vitamin b ka atyadhik sevan hanikarak hai?")

class RAGSource(BaseModel):
    chunk_id: str
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

@router.post("/query", response_model=RAGQueryResponse, summary="Perform Grounded RAG Query")
def rag_query_endpoint(request: RAGQueryRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty or whitespace-only."
        )

    try:
        service = get_rag_service()
        response_data = service.answer(query=request.query.strip())
        return response_data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG execution error: {str(e)}"
        )
