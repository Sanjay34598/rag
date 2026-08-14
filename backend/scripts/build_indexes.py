import os
import sys
import json
import time
import pyarrow.parquet as pq
import pandas as pd
from typing import List, Dict, Any, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings, DATA_DIR
from app.services.retrieval.embeddings import get_embedding_service
from app.services.retrieval.vector_store import FAISSVectorStore
from app.services.retrieval.bm25_retriever import BM25Retriever

def extract_chunks_from_dataset(dataset_path: str, max_chunks: int = 1500) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    print(f"[BuildIndexes] Loading dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")

    pf = pq.ParquetFile(dataset_path)
    cols = ["query_id", "query", "Eng_Query", "target_lang", "passages"]
    
    chunks = []
    eval_queries = []
    
    try:
        for batch in pf.iter_batches(batch_size=100, columns=cols):
            df = batch.to_pandas()
            for _, row in df.iterrows():
                qid = row["query_id"]
                qtext = row.get("query") or row.get("Eng_Query", "")
                target_lang = row.get("target_lang", "hin_Deva")
                passages_dict = row.get("passages", {})
                
                trans_passages = passages_dict.get("Translated_passages", [])
                eng_passages = passages_dict.get("English_passages", [])
                is_selected = passages_dict.get("is_selected", [])
                
                passage_list = trans_passages if len(trans_passages) > 0 else eng_passages
                
                eval_queries.append({
                    "query_id": qid,
                    "query": qtext,
                    "target_lang": target_lang
                })
                
                for p_idx, text in enumerate(passage_list):
                    if not text or not str(text).strip():
                        continue
                    clean_text = str(text).strip()
                    chunk_id = f"p_{qid}_{p_idx}"
                    rel_label = int(is_selected[p_idx]) if p_idx < len(is_selected) else 0
                    
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": clean_text,
                        "query_id": qid,
                        "passage_index": p_idx,
                        "is_selected": rel_label,
                        "target_lang": target_lang
                    })
                    if len(chunks) >= max_chunks:
                        break
                if len(chunks) >= max_chunks:
                    break
            if len(chunks) >= max_chunks:
                break
    except Exception as e:
        print(f"[BuildIndexes] Batch reading completed: {e}")
        
    print(f"[BuildIndexes] Successfully extracted {len(chunks)} chunks across {len(eval_queries)} queries.")
    return chunks, eval_queries

def main():
    start_time = time.time()
    max_c = int(os.getenv("MAX_CHUNKS_TO_INDEX", "1500"))
    print("==================================================")
    print(f"STAGE 3: BUILDING HYBRID RETRIEVAL INDEXES ({max_c} chunks limit)")
    print("==================================================")
    
    chunks, eval_queries = extract_chunks_from_dataset(settings.DATASET_PATH, max_chunks=max_c)
    
    os.makedirs(os.path.dirname(settings.PROCESSED_CHUNKS_PATH), exist_ok=True)
    with open(settings.PROCESSED_CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    eval_queries_path = os.path.join(DATA_DIR, "eval_queries.json")
    with open(eval_queries_path, "w", encoding="utf-8") as f:
        json.dump(eval_queries, f, ensure_ascii=False, indent=2)
        
    print("\n--- Step 2: Building FAISS Dense Index ---")
    embed_service = get_embedding_service()
    texts = [c["text"] for c in chunks]
    
    t0 = time.time()
    embeddings = embed_service.encode(texts, normalize=True, batch_size=256, show_progress=False)
    print(f"[BuildIndexes] Generated embeddings shape: {embeddings.shape} in {time.time() - t0:.2f}s")
    
    vector_store = FAISSVectorStore()
    vector_store.build_index(embeddings, chunks)
    vector_store.save()

    print("\n--- Step 3: Building BM25 Sparse Index ---")
    t0 = time.time()
    bm25 = BM25Retriever()
    bm25.build_index(chunks)
    bm25.save()
    print(f"[BuildIndexes] Built BM25 index in {time.time() - t0:.2f}s")

    faiss_size_mb = os.path.getsize(settings.FAISS_INDEX_PATH) / (1024 * 1024)
    bm25_size_mb = os.path.getsize(settings.BM25_INDEX_PATH) / (1024 * 1024)
    chunks_size_mb = os.path.getsize(settings.PROCESSED_CHUNKS_PATH) / (1024 * 1024)
    
    total_time = time.time() - start_time
    print("\n==================================================")
    print("INDEX BUILDING COMPLETE STATISTICS")
    print("==================================================")
    print(f"Total Chunks Indexed:  {len(chunks)}")
    print(f"Embedding Model:      {settings.EMBEDDING_MODEL}")
    print(f"FAISS Index File:     {settings.FAISS_INDEX_PATH} ({faiss_size_mb:.2f} MB)")
    print(f"BM25 Index File:      {settings.BM25_INDEX_PATH} ({bm25_size_mb:.2f} MB)")
    print(f"Total Pipeline Time:  {total_time:.2f} seconds")
    print("==================================================")

if __name__ == "__main__":
    main()
