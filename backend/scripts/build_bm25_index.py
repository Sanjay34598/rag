import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings
from app.services.retrieval.bm25_retriever import BM25Retriever

def main():
    print("[build_bm25_index] Starting BM25 Index Creation...")
    if not os.path.exists(settings.PROCESSED_CHUNKS_PATH):
        raise FileNotFoundError(f"Processed chunks not found at {settings.PROCESSED_CHUNKS_PATH}. Run build_indexes.py first.")
        
    with open(settings.PROCESSED_CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    print(f"[build_bm25_index] Loaded {len(chunks)} chunks.")
    t0 = time.time()
    bm25 = BM25Retriever()
    bm25.build_index(chunks)
    bm25.save()
    print(f"[build_bm25_index] Built BM25 index in {time.time() - t0:.2f}s")

if __name__ == "__main__":
    main()
