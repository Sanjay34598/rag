import os
from pathlib import Path

# Suppress TensorFlow dependency conflicts in transformers
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

class Settings:
    PROJECT_NAME: str = "Voice-Enabled RAG System"
    VERSION: str = "0.4.1"
    API_V1_STR: str = "/api/v1"
    
    # Dataset & Index Paths
    DATASET_PATH: str = os.getenv("DATASET_PATH", str(DATA_DIR / "sample_hinval.parquet"))
    PROCESSED_CHUNKS_PATH: str = os.getenv("PROCESSED_CHUNKS_PATH", str(DATA_DIR / "processed_chunks.json"))
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", str(DATA_DIR / "faiss_index.bin"))
    FAISS_METADATA_PATH: str = os.getenv("FAISS_METADATA_PATH", str(DATA_DIR / "faiss_metadata.json"))
    BM25_INDEX_PATH: str = os.getenv("BM25_INDEX_PATH", str(DATA_DIR / "bm25_index.pkl"))
    
    # Retrieval Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    DENSE_WEIGHT: float = float(os.getenv("DENSE_WEIGHT", "0.7"))
    BM25_WEIGHT: float = float(os.getenv("BM25_WEIGHT", "0.3"))
    CANDIDATE_K: int = int(os.getenv("CANDIDATE_K", "20"))
    TOP_K: int = int(os.getenv("TOP_K", "5"))
    
    # Reranker Settings
    RERANKER_ENABLED: bool = os.getenv("RERANKER_ENABLED", "false").lower() in ("true", "1", "yes")
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    RERANKER_TOP_N: int = int(os.getenv("RERANKER_TOP_N", "5"))

    # LLM Settings (Stage 4)
    LLM_MODE: str = os.getenv("LLM_MODE", "fallback").lower()  # "real" or "fallback"
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-1.5-flash")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", os.getenv("GEMINI_API_KEY", os.getenv("OPENAI_API_KEY", "")))
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "2"))
    LLM_TIMEOUT: float = float(os.getenv("LLM_TIMEOUT", "10.0"))

    # Guardrails Settings
    MIN_RETRIEVAL_SCORE: float = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.2"))
    MIN_CONTEXT_CHUNKS: int = int(os.getenv("MIN_CONTEXT_CHUNKS", "1"))
    MAX_CONTEXT_CHUNKS: int = int(os.getenv("MAX_CONTEXT_CHUNKS", "5"))

settings = Settings()
