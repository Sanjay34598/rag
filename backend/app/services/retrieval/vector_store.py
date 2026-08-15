import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from app.core.config import settings

class FAISSVectorStore:
    def __init__(self, index_path: str = None, metadata_path: str = None):
        self.index_path = index_path or settings.FAISS_INDEX_PATH
        self.metadata_path = metadata_path or settings.FAISS_METADATA_PATH
        self.index: Any = None
        self.metadata: List[Dict[str, Any]] = []
        self._is_loaded = False

    def build_index(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        import faiss
        if len(embeddings) != len(metadata):
            raise ValueError(f"Embeddings count ({len(embeddings)}) must match metadata count ({len(metadata)})")

        dimension = embeddings.shape[1]
        print(f"[FAISSVectorStore] Building IndexFlatIP with dimension={dimension}, items={len(embeddings)}")
        
        # Ensure float32 and normalized
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        self.metadata = metadata
        self._is_loaded = True

    def save(self) -> None:
        import faiss
        if self.index is None:
            raise ValueError("Cannot save empty FAISS index.")
        
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            
        print(f"[FAISSVectorStore] Index saved to {self.index_path} ({self.index.ntotal} vectors)")

    def load(self) -> bool:
        import faiss
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            print(f"[FAISSVectorStore] Index or metadata missing at {self.index_path}")
            return False
        
        try:
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            self._is_loaded = True
            print(f"[FAISSVectorStore] Loaded index with {self.index.ntotal} vectors from {self.index_path}")
            return True
        except Exception as e:
            print(f"[FAISSVectorStore] Error loading FAISS index: {e}")
            return False

    def search(self, query_vector: np.ndarray, top_k: int = 20) -> List[Dict[str, Any]]:
        import faiss
        if not self._is_loaded or self.index is None:
            raise RuntimeError("FAISS index is not loaded.")
        
        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)
            
        query_vector = np.ascontiguousarray(query_vector.astype(np.float32))
        faiss.normalize_L2(query_vector)

        actual_k = min(top_k, self.index.ntotal)
        if actual_k == 0:
            return []

        scores, indices = self.index.search(query_vector, actual_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            results.append({
                "chunk_id": meta.get("chunk_id", str(idx)),
                "text": meta.get("text", ""),
                "dense_score": float(score),
                "metadata": meta.get("metadata", meta)
            })
        return results

    def is_loaded(self) -> bool:
        return self._is_loaded and self.index is None or (self.index is not None and self._is_loaded)
