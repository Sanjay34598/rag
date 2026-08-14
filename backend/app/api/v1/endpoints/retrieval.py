from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from app.services.retrieval.retrieval_service import get_retrieval_service

router = APIRouter()

class RetrieveRequest(BaseModel):
    query: str = Field(..., description="User search query string", example="kya vitamin b ka atyadhik sevan hanikarak hai?")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of final results to return")

class LatencyBreakdown(BaseModel):
    embedding_ms: float
    faiss_ms: float
    bm25_ms: float
    fusion_ms: float
    reranking_ms: float

class ChunkResult(BaseModel):
    chunk_id: str
    text: str
    score: float
    dense_score: float
    bm25_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RetrieveResponse(BaseModel):
    query: str
    results: List[ChunkResult]
    reranking_enabled: bool = False
    latency_ms: float
    latency_breakdown: LatencyBreakdown

@router.post("/retrieve", response_model=RetrieveResponse, summary="Perform Hybrid Retrieval")
def retrieve_endpoint(request: RetrieveRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty or whitespace-only."
        )
        
    try:
        service = get_retrieval_service()
        response_data = service.retrieve(query=request.query.strip(), top_k=request.top_k)
        return response_data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval error: {str(e)}"
        )
