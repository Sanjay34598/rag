import os
import torch
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from app.core.config import settings

class EmbeddingService:
    _instance = None

    def __new__(cls, model_name: str = None):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = None):
        if self._initialized:
            return
        
        os.environ["USE_TF"] = "0"
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        
        self.model_name = model_name or settings.EMBEDDING_MODEL
        print(f"[EmbeddingService] Initializing embedding model: {self.model_name}")
        
        # Optimize CPU threads for fast batch encoding
        num_threads = min(8, os.cpu_count() or 4)
        torch.set_num_threads(num_threads)
        
        self.model = SentenceTransformer(self.model_name)
        # Warmup pass to pre-allocate PyTorch CPU memory and initialize oneDNN threads
        with torch.inference_mode():
            self.model.encode(["नमस्ते दुनिया warmup"], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        self._initialized = True
        print(f"[EmbeddingService] Embedding model loaded and pre-warmed (CPU Threads: {num_threads}).")

    def encode(self, texts: Union[str, List[str]], normalize: bool = True, batch_size: int = 256, show_progress: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
            
        with torch.inference_mode():
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=normalize
            )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        return self.encode([query], normalize=normalize, show_progress=False)[0]

def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
