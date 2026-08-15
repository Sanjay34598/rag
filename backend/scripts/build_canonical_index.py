import os
import sys
import json
import time
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd
from typing import List, Dict, Any, Tuple
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings, DATA_DIR
from app.services.retrieval.embeddings import get_embedding_service
from app.services.retrieval.vector_store import FAISSVectorStore
from app.services.retrieval.bm25_retriever import BM25Retriever

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def get_parquet_paths() -> Tuple[str, str]:
    hin_path = os.path.join(DATA_DIR, "sample_hinval.parquet")
    if not os.path.exists(hin_path):
        print(f"[BuildCanonical] Downloading validation/hinval.parquet from HF...", flush=True)
        hin_path = hf_hub_download(repo_id="ai4bharat/MSMARCO-XI", filename="validation/hinval.parquet", repo_type="dataset")

    tel_path = os.path.join(DATA_DIR, "sample_telval.parquet")
    if not os.path.exists(tel_path):
        cache_path = os.path.expanduser(r"~\.cache\huggingface\hub\datasets--ai4bharat--MSMARCO-XI\snapshots\bf5cdc1f26e581e519018e434db14edd1b77602b\validation\telval.parquet")
        if os.path.exists(cache_path):
            tel_path = cache_path
        else:
            print(f"[BuildCanonical] Locating validation/telval.parquet...", flush=True)
            try:
                tel_path = hf_hub_download(repo_id="ai4bharat/MSMARCO-XI", filename="validation/telval.parquet", repo_type="dataset")
            except Exception as e:
                print(f"[BuildCanonical] Warning: HF download error for telval.parquet: {e}", flush=True)
                tel_path = hin_path
    return hin_path, tel_path

def build_canonical_chunks(max_chunks: int = 1500) -> List[Dict[str, Any]]:
    hin_path, tel_path = get_parquet_paths()
    print(f"[BuildCanonical] Using dataset source (hinval): {hin_path}", flush=True)
    print(f"[BuildCanonical] Using dataset source (telval): {tel_path}", flush=True)

    # First pass: map telugu queries by query_id from telval
    tel_queries_map = {}
    try:
        pf_tel = pq.ParquetFile(tel_path)
        for batch in pf_tel.iter_batches(batch_size=100, columns=["query_id", "query"]):
            df = batch.to_pandas()
            for _, row in df.iterrows():
                qid = int(row["query_id"])
                q_te = str(row.get("query") or "").strip()
                if qid and q_te:
                    tel_queries_map[qid] = q_te
    except Exception as e:
        print(f"[BuildCanonical] Warning reading telugu queries: {e}", flush=True)

    canonical_chunks = []
    query_id_set = set()
    pf_hin = pq.ParquetFile(hin_path)
    cols = ["query_id", "query", "Eng_Query", "source_lang", "target_lang", "passages"]

    print("[BuildCanonical] Extracting canonical English evidence chunks from MSMARCO-XI...", flush=True)
    for batch in pf_hin.iter_batches(batch_size=100, columns=cols):
        df = batch.to_pandas()
        for _, row in df.iterrows():
            qid = int(row["query_id"])
            q_hi = str(row.get("query") or "").strip()
            q_en = str(row.get("Eng_Query") or "").strip()
            q_te = tel_queries_map.get(qid, "")
            source_lang = str(row.get("source_lang") or "eng_Latn").strip()
            passages = row.get("passages", {})

            if not isinstance(passages, dict):
                continue

            eng_passages = passages.get("English_passages", [])
            is_selected = passages.get("is_selected", [])

            for p_idx, text in enumerate(eng_passages):
                if not text or not str(text).strip():
                    continue
                rel_label = int(is_selected[p_idx]) if p_idx < len(is_selected) else 0
                
                canonical_chunks.append({
                    "chunk_id": f"p_{qid}_{p_idx}",
                    "query_id": qid,
                    "text": str(text).strip(),
                    "language": "en",
                    "source_lang": source_lang,
                    "target_lang": None,
                    "source_type": "original_english",
                    "original_query": q_en,
                    "translated_queries": {
                        "hi": q_hi,
                        "te": q_te
                    },
                    "score": 0.0,
                    "passage_index": p_idx,
                    "is_selected": rel_label
                })
                if len(canonical_chunks) >= max_chunks:
                    break
            if len(canonical_chunks) >= max_chunks:
                break
        if len(canonical_chunks) >= max_chunks:
            break

    print(f"[BuildCanonical] Total canonical English evidence chunks extracted: {len(canonical_chunks)}", flush=True)
    return canonical_chunks

def main():
    start_time = time.time()
    max_c = int(os.getenv("MAX_CANONICAL_CHUNKS", "1500"))
    print("==================================================", flush=True)
    print(f"BUILDING CANONICAL RETRIEVAL INDEX ({max_c} chunks)", flush=True)
    print("==================================================", flush=True)

    canonical_chunks = build_canonical_chunks(max_chunks=max_c)
    embed_service = get_embedding_service()

    canonical_dir = DATA_DIR / "indexes" / "canonical"
    os.makedirs(canonical_dir, exist_ok=True)

    # Save processed chunks
    processed_path = canonical_dir / "processed_chunks.json"
    with open(processed_path, "w", encoding="utf-8") as f:
        json.dump(canonical_chunks, f, ensure_ascii=False, indent=2)

    # Generate Dense Vector Embeddings
    texts = [c["text"] for c in canonical_chunks]
    t0 = time.time()
    embeddings = embed_service.encode(texts, normalize=True, batch_size=256, show_progress=False)
    print(f"Encrypted/Encoded {len(texts)} canonical chunks (shape: {embeddings.shape}) in {time.time() - t0:.2f}s", flush=True)

    # Build & Save FAISS Index
    faiss_store = FAISSVectorStore()
    faiss_store.index_path = str(canonical_dir / "faiss_index.bin")
    faiss_store.metadata_path = str(canonical_dir / "faiss_metadata.json")
    faiss_store.build_index(embeddings, canonical_chunks)
    faiss_store.save()

    # Build & Save BM25 Index
    bm25_retriever = BM25Retriever()
    bm25_retriever.index_path = str(canonical_dir / "bm25_index.pkl")
    bm25_retriever.build_index(canonical_chunks)
    bm25_retriever.save()

    faiss_size_mb = os.path.getsize(canonical_dir / "faiss_index.bin") / (1024 * 1024)
    bm25_size_mb = os.path.getsize(canonical_dir / "bm25_index.pkl") / (1024 * 1024)

    total_time = time.time() - start_time
    print("\n==================================================", flush=True)
    print("CANONICAL INDEX BUILDING COMPLETE", flush=True)
    print("==================================================", flush=True)
    print(f"Directory:  {canonical_dir}", flush=True)
    print(f"Chunks:     {len(canonical_chunks)}", flush=True)
    print(f"FAISS Size: {faiss_size_mb:.2f} MB", flush=True)
    print(f"BM25 Size:  {bm25_size_mb:.2f} MB", flush=True)
    print(f"Total Time: {total_time:.2f}s", flush=True)
    print("==================================================", flush=True)

if __name__ == "__main__":
    main()
