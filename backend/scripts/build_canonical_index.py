import os
import sys
import json
import time
from pathlib import Path
import pyarrow.parquet as pq
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
    hin_path = os.path.expanduser(r"~\.cache\huggingface\hub\datasets--ai4bharat--MSMARCO-XI\snapshots\bf5cdc1f26e581e519018e434db14edd1b77602b\validation\hinval.parquet")
    if not os.path.exists(hin_path):
        print(f"[BuildCanonical] Downloading validation/hinval.parquet from HF...", flush=True)
        hin_path = hf_hub_download(repo_id="ai4bharat/MSMARCO-XI", filename="validation/hinval.parquet", repo_type="dataset")

    tel_path = os.path.expanduser(r"~\.cache\huggingface\hub\datasets--ai4bharat--MSMARCO-XI\snapshots\bf5cdc1f26e581e519018e434db14edd1b77602b\validation\telval.parquet")
    if not os.path.exists(tel_path):
        print(f"[BuildCanonical] Downloading validation/telval.parquet from HF...", flush=True)
        tel_path = hf_hub_download(repo_id="ai4bharat/MSMARCO-XI", filename="validation/telval.parquet", repo_type="dataset")
    return hin_path, tel_path

def build_canonical_chunks(target_chunks: int = 15000) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    hin_path, tel_path = get_parquet_paths()
    print(f"[BuildCanonical] Dataset hinval path: {hin_path}", flush=True)
    print(f"[BuildCanonical] Dataset telval path: {tel_path}", flush=True)

    # 1. Map telugu queries by query_id from telval
    tel_queries_map = {}
    try:
        pf_tel = pq.ParquetFile(tel_path)
        print(f"[BuildCanonical] Reading Telugu query mappings from {tel_path}...", flush=True)
        table_tel = pf_tel.read(columns=["query_id", "query"])
        df_tel = table_tel.to_pandas()
        for _, row in df_tel.iterrows():
            qid = int(row["query_id"])
            q_te = str(row.get("query") or "").strip()
            if qid and q_te and qid not in tel_queries_map:
                tel_queries_map[qid] = q_te
        print(f"[BuildCanonical] Mapped {len(tel_queries_map)} Telugu query translations.", flush=True)
    except Exception as e:
        print(f"[BuildCanonical] Warning reading telugu queries: {e}", flush=True)

    canonical_chunks = []
    seen_texts = set()
    unique_qids = set()
    duplicates_count = 0

    pf_hin = pq.ParquetFile(hin_path)
    cols = ["query_id", "query", "Eng_Query", "source_lang", "target_lang", "passages"]

    print(f"[BuildCanonical] Extracting up to {target_chunks} unique English evidence chunks...", flush=True)
    table_hin = pf_hin.read(columns=cols)
    df_hin = table_hin.to_pandas()

    for _, row in df_hin.iterrows():
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
            
            clean_text = str(text).strip()
            text_key = clean_text.lower()

            if text_key in seen_texts:
                duplicates_count += 1
                continue

            seen_texts.add(text_key)
            unique_qids.add(qid)
            rel_label = int(is_selected[p_idx]) if p_idx < len(is_selected) else 0

            canonical_chunks.append({
                "chunk_id": f"p_{qid}_{p_idx}",
                "query_id": qid,
                "text": clean_text,
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

            if len(canonical_chunks) >= target_chunks:
                break
        if len(canonical_chunks) >= target_chunks:
            break

    stats = {
        "indexed_passages": len(canonical_chunks),
        "unique_query_ids": len(unique_qids),
        "duplicates_removed": duplicates_count
    }
    print(f"[BuildCanonical] Extraction complete: {stats['indexed_passages']} passages across {stats['unique_query_ids']} unique queries ({stats['duplicates_removed']} duplicates removed).", flush=True)
    return canonical_chunks, stats

def main():
    start_time = time.time()
    target_c = int(os.getenv("MAX_CANONICAL_CHUNKS", "15000"))
    print("==================================================", flush=True)
    print(f"BUILDING CANONICAL RETRIEVAL INDEX ({target_c} passages)", flush=True)
    print("==================================================", flush=True)

    canonical_chunks, stats = build_canonical_chunks(target_chunks=target_c)
    embed_service = get_embedding_service()

    canonical_dir = DATA_DIR / "indexes" / "canonical"
    os.makedirs(canonical_dir, exist_ok=True)

    # Save processed chunks
    processed_path = canonical_dir / "processed_chunks.json"
    t0 = time.time()
    with open(processed_path, "w", encoding="utf-8") as f:
        json.dump(canonical_chunks, f, ensure_ascii=False, indent=2)
    processed_size_mb = os.path.getsize(processed_path) / (1024 * 1024)

    # Generate Dense Vector Embeddings
    texts = [c["text"] for c in canonical_chunks]
    print(f"[BuildCanonical] Generating dense embeddings for {len(texts)} chunks...", flush=True)
    import torch
    torch.set_num_threads(os.cpu_count() or 8)
    t0 = time.time()
    embeddings = embed_service.encode(texts, normalize=True, batch_size=64, show_progress=True)
    print(f"[BuildCanonical] Dense embeddings generated (shape: {embeddings.shape}) in {time.time() - t0:.2f}s", flush=True)

    # Build & Save FAISS Index
    print("[BuildCanonical] Building FAISS vector store...", flush=True)
    faiss_store = FAISSVectorStore()
    faiss_store.index_path = str(canonical_dir / "faiss_index.bin")
    faiss_store.metadata_path = str(canonical_dir / "faiss_metadata.json")
    faiss_store.build_index(embeddings, canonical_chunks)
    faiss_store.save()

    # Build & Save BM25 Index
    print("[BuildCanonical] Building BM25 index...", flush=True)
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
    print(f"Directory:             {canonical_dir}", flush=True)
    print(f"Indexed Passages:      {stats['indexed_passages']}", flush=True)
    print(f"Unique Query IDs:      {stats['unique_query_ids']}", flush=True)
    print(f"Duplicates Removed:    {stats['duplicates_removed']}", flush=True)
    print(f"processed_chunks.json: {processed_size_mb:.2f} MB", flush=True)
    print(f"FAISS Index Size:      {faiss_size_mb:.2f} MB", flush=True)
    print(f"BM25 Index Size:       {bm25_size_mb:.2f} MB", flush=True)
    print(f"Total Build Time:      {total_time:.2f}s", flush=True)
    print("==================================================", flush=True)

if __name__ == "__main__":
    main()
