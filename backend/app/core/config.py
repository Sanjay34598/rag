import os
from pathlib import Path
from dotenv import load_dotenv

# Suppress TensorFlow dependency conflicts in transformers
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

# Load .env file from project root or parent directory
env_path = BASE_DIR / ".env"
if not env_path.exists() and (BASE_DIR.parent / ".env").exists():
    env_path = BASE_DIR.parent / ".env"

load_dotenv(env_path, override=True)

class Settings:
    PROJECT_NAME: str = "Voice-Enabled RAG System"
    VERSION: str = "0.4.1"
    API_V1_STR: str = "/api/v1"
    
    # Dataset & Index Paths
    INDEXES_DIR: Path = DATA_DIR / "indexes"
    CANONICAL_INDEX_DIR: Path = DATA_DIR / "indexes" / "canonical"
    DATASET_PATH: str = os.getenv("DATASET_PATH", str(DATA_DIR / "sample_hinval.parquet"))
    PROCESSED_CHUNKS_PATH: str = os.getenv("PROCESSED_CHUNKS_PATH", str(CANONICAL_INDEX_DIR / "processed_chunks.json"))
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", str(CANONICAL_INDEX_DIR / "faiss_index.bin"))
    FAISS_METADATA_PATH: str = os.getenv("FAISS_METADATA_PATH", str(CANONICAL_INDEX_DIR / "faiss_metadata.json"))
    BM25_INDEX_PATH: str = os.getenv("BM25_INDEX_PATH", str(CANONICAL_INDEX_DIR / "bm25_index.pkl"))
    
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

    # LLM Settings (Groq Exclusive)
    LLM_MODE: str = os.getenv("LLM_MODE", "real").lower()  # "real" or "fallback"
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq").lower()
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    LLM_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "0"))
    LLM_TIMEOUT: float = float(os.getenv("LLM_TIMEOUT", "10.0"))

    # Guardrails Settings
    MIN_RETRIEVAL_SCORE: float = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.2"))
    MIN_CONTEXT_CHUNKS: int = int(os.getenv("MIN_CONTEXT_CHUNKS", "1"))
    MAX_CONTEXT_CHUNKS: int = int(os.getenv("MAX_CONTEXT_CHUNKS", "5"))

    # Sarvam STT Settings (Stage 5B)
    SARVAM_API_KEY: str = os.getenv("SARVAM_API_KEY", "")
    SARVAM_STT_MODEL: str = os.getenv("SARVAM_STT_MODEL", "saaras:v3")
    SARVAM_STT_MODE: str = os.getenv("SARVAM_STT_MODE", "transcribe")
    SARVAM_LANGUAGE_CODE: str = os.getenv("SARVAM_LANGUAGE_CODE", "hi-IN")
    SARVAM_STT_URL: str = os.getenv("SARVAM_STT_URL", "https://api.sarvam.ai/speech-to-text")
    SARVAM_TIMEOUT: float = float(os.getenv("SARVAM_TIMEOUT", "15.0"))

    # CORS Settings
    ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173")

settings = Settings()

if settings.LLM_PROVIDER != "groq":
    raise ValueError("LLM_PROVIDER must be 'groq'. Gemini and other providers are completely disabled.")

if settings.LLM_MODE == "real" and not settings.GROQ_API_KEY:
    print("[WARNING] GROQ_API_KEY is not configured.")

