from fastapi import APIRouter
from app.api.v1.endpoints import retrieval, rag

api_router = APIRouter()
api_router.include_router(retrieval.router, prefix="", tags=["retrieval"])
api_router.include_router(rag.router, prefix="/rag", tags=["rag"])
