import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings
from app.services.retrieval.embeddings import get_embedding_service
from app.services.retrieval.vector_store import FAISSVectorStore

def main():
    print("[build_vector_index] Starting FAISS Index Creation...")
    if not os.path.exists(settings.PROCESSED_CHUNKS_PATH):
        raise FileNotFoundError(f"Processed chunks not found at {settings.PROCESSED_CHUNKS_PATH}. Run build_indexes.py first.")
        
    with open(settings.PROCESSED_CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    print(f"[build_vector_index] Loaded {len(chunks)} chunks.")
    embed_service = get_embedding_service()
    texts = [c["text"] for c in chunks]
    
    t0 = time.time()
    embeddings = embed_service.encode(texts, normalize=True, batch_size=128)
    print(f"[build_vector_index] Generated embeddings in {time.time() - t0:.2f}s")
    
    store = FAISSVectorStore()
    store.build_index(embeddings, chunks)
    store.save()
    print("[build_vector_index] FAISS vector store build complete.")

if __name__ == "__main__":
    main()
