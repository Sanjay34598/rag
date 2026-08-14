import os
import re
import pickle
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from app.core.config import settings

def tokenize_text(text: str) -> List[str]:
    """
    Multilingual word tokenizer preserving Devanagari and English alphanumeric words.
    """
    if not text:
        return []
    # Lowercase and split on whitespace / non-alphanumeric punctuation except Devanagari block (\u0900-\u097F)
    text = text.lower()
    tokens = re.findall(r'[\w\u0900-\u097F]+', text)
    return tokens

class BM25Retriever:
    def __init__(self, index_path: str = None):
        self.index_path = index_path or settings.BM25_INDEX_PATH
        self.bm25: BM25Okapi = None
        self.metadata: List[Dict[str, Any]] = []
        self.corpus_tokens: List[List[str]] = []
        self._is_loaded = False

    def build_index(self, metadata: List[Dict[str, Any]]) -> None:
        print(f"[BM25Retriever] Tokenizing {len(metadata)} chunks...")
        self.metadata = metadata
        self.corpus_tokens = [tokenize_text(m.get("text", "")) for m in metadata]
        
        print(f"[BM25Retriever] Building BM25Okapi index...")
        self.bm25 = BM25Okapi(self.corpus_tokens)
        self._is_loaded = True

    def save(self) -> None:
        if self.bm25 is None:
            raise ValueError("Cannot save empty BM25 index.")
        
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        data = {
            "bm25": self.bm25,
            "metadata": self.metadata,
            "corpus_tokens": self.corpus_tokens
        }
        with open(self.index_path, "wb") as f:
            pickle.dump(data, f)
        print(f"[BM25Retriever] Saved BM25 index to {self.index_path} ({len(self.metadata)} items)")

    def load(self) -> bool:
        if not os.path.exists(self.index_path):
            print(f"[BM25Retriever] Index missing at {self.index_path}")
            return False
        
        try:
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.metadata = data["metadata"]
            self.corpus_tokens = data.get("corpus_tokens", [])
            self._is_loaded = True
            print(f"[BM25Retriever] Loaded BM25 index from {self.index_path} ({len(self.metadata)} items)")
            return True
        except Exception as e:
            print(f"[BM25Retriever] Error loading BM25 index: {e}")
            return False

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        if not self._is_loaded or self.bm25 is None:
            raise RuntimeError("BM25 index is not loaded.")
        
        tokens = tokenize_text(query)
        if not tokens:
            return []
        
        scores = self.bm25.get_scores(tokens)
        
        # Get top K indices
        actual_k = min(top_k, len(scores))
        if actual_k == 0:
            return []

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:actual_k]
        
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            meta = self.metadata[idx]
            results.append({
                "chunk_id": meta.get("chunk_id", str(idx)),
                "text": meta.get("text", ""),
                "bm25_score": score,
                "metadata": meta.get("metadata", meta)
            })
        return results
