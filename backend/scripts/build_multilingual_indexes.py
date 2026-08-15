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
        print(f"[BuildMultilingual] Downloading validation/hinval.parquet from HF...", flush=True)
        hin_path = hf_hub_download(repo_id="ai4bharat/MSMARCO-XI", filename="validation/hinval.parquet", repo_type="dataset")

    tel_path = os.path.join(DATA_DIR, "sample_telval.parquet")
    if not os.path.exists(tel_path):
        cache_path = os.path.expanduser(r"~\.cache\huggingface\hub\datasets--ai4bharat--MSMARCO-XI\snapshots\bf5cdc1f26e581e519018e434db14edd1b77602b\validation\telval.parquet")
        if os.path.exists(cache_path):
            tel_path = cache_path
        else:
            print(f"[BuildMultilingual] Locating validation/telval.parquet...", flush=True)
            try:
                tel_path = hf_hub_download(repo_id="ai4bharat/MSMARCO-XI", filename="validation/telval.parquet", repo_type="dataset")
            except Exception as e:
                print(f"[BuildMultilingual] Warning: HF download error for telval.parquet: {e}", flush=True)
                tel_path = hin_path
    return hin_path, tel_path

def build_multilingual_chunks(max_chunks_per_lang: int = 1500) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    hin_path, tel_path = get_parquet_paths()
    print(f"[BuildMultilingual] Using Hindi source: {hin_path}", flush=True)
    print(f"[BuildMultilingual] Using Telugu source: {tel_path}", flush=True)

    pf_hin = pq.ParquetFile(hin_path)
    cols = ["query_id", "query", "Eng_Query", "source_lang", "target_lang", "passages"]

    target_query_ids = []
    query_id_set = set()
    
    hi_chunks = []
    en_chunks = []
    eval_queries = {"en": [], "hi": [], "te": []}

    print("[BuildMultilingual] Extracting Hindi & English chunks...", flush=True)
    for batch in pf_hin.iter_batches(batch_size=100, columns=cols):
        df = batch.to_pandas()
        for _, row in df.iterrows():
            qid = int(row["query_id"])
            q_hi = str(row.get("query") or "").strip()
            q_en = str(row.get("Eng_Query") or "").strip()
            source_lang = str(row.get("source_lang") or "eng_Latn").strip()
            target_lang_hi = str(row.get("target_lang") or "hin_Deva").strip()
            passages = row.get("passages", {})

            if not isinstance(passages, dict):
                continue

            trans_passages = passages.get("Translated_passages", [])
            eng_passages = passages.get("English_passages", [])
            is_selected = passages.get("is_selected", [])

            if qid not in query_id_set:
                query_id_set.add(qid)
                target_query_ids.append(qid)

            # Build Hindi chunks
            if len(hi_chunks) < max_chunks_per_lang:
                eval_queries["hi"].append({"query_id": qid, "query": q_hi or q_en, "target_lang": target_lang_hi})
                for p_idx, text in enumerate(trans_passages):
                    if not text or not str(text).strip():
                        continue
                    rel_label = int(is_selected[p_idx]) if p_idx < len(is_selected) else 0
                    hi_chunks.append({
                        "chunk_id": f"p_{qid}_{p_idx}",
                        "language": "hi",
                        "query_id": qid,
                        "passage_index": p_idx,
                        "text": str(text).strip(),
                        "score": 0.0,
                        "source_lang": source_lang,
                        "target_lang": target_lang_hi,
                        "is_selected": rel_label
                    })
                    if len(hi_chunks) >= max_chunks_per_lang:
                        break

            # Build English chunks
            if len(en_chunks) < max_chunks_per_lang:
                eval_queries["en"].append({"query_id": qid, "query": q_en or q_hi, "target_lang": "eng_Latn"})
                for p_idx, text in enumerate(eng_passages):
                    if not text or not str(text).strip():
                        continue
                    rel_label = int(is_selected[p_idx]) if p_idx < len(is_selected) else 0
                    en_chunks.append({
                        "chunk_id": f"p_{qid}_{p_idx}",
                        "language": "en",
                        "query_id": qid,
                        "passage_index": p_idx,
                        "text": str(text).strip(),
                        "score": 0.0,
                        "source_lang": source_lang,
                        "target_lang": "eng_Latn",
                        "is_selected": rel_label
                    })
                    if len(en_chunks) >= max_chunks_per_lang:
                        break

            if len(hi_chunks) >= max_chunks_per_lang and len(en_chunks) >= max_chunks_per_lang:
                break
        if len(hi_chunks) >= max_chunks_per_lang and len(en_chunks) >= max_chunks_per_lang:
            break

    print(f"[BuildMultilingual] Hindi chunks: {len(hi_chunks)}, English chunks: {len(en_chunks)}", flush=True)

    # 2. Second pass: Extract Telugu chunks for matching query IDs from Telugu Parquet
    print("[BuildMultilingual] Extracting Telugu chunks (matching query_ids)...", flush=True)
    te_chunks = []
    pf_tel = pq.ParquetFile(tel_path)
    target_qid_set = set(target_query_ids)

    for batch in pf_tel.iter_batches(batch_size=100, columns=cols):
        df = batch.to_pandas()
        for _, row in df.iterrows():
            qid = int(row["query_id"])
            if qid not in target_qid_set and len(te_chunks) >= max_chunks_per_lang:
                continue

            q_te = str(row.get("query") or "").strip()
            source_lang = str(row.get("source_lang") or "eng_Latn").strip()
            # READ ACTUAL target_lang FROM DATASET ROW - DO NOT HARDCODE
            target_lang_te = str(row.get("target_lang") or "tel_Telu").strip()
            passages = row.get("passages", {})

            if not isinstance(passages, dict):
                continue

            trans_passages = passages.get("Translated_passages", [])
            eng_passages = passages.get("English_passages", [])
            is_selected = passages.get("is_selected", [])

            passage_list = trans_passages if len(trans_passages) > 0 else eng_passages

            eval_queries["te"].append({"query_id": qid, "query": q_te, "target_lang": target_lang_te})

            for p_idx, text in enumerate(passage_list):
                if not text or not str(text).strip():
                    continue
                rel_label = int(is_selected[p_idx]) if p_idx < len(is_selected) else 0
                te_chunks.append({
                    "chunk_id": f"p_{qid}_{p_idx}",
                    "language": "te",
                    "query_id": qid,
                    "passage_index": p_idx,
                    "text": str(text).strip(),
                    "score": 0.0,
                    "source_lang": source_lang,
                    "target_lang": target_lang_te,
                    "is_selected": rel_label
                })
                if len(te_chunks) >= max_chunks_per_lang:
                    break
            if len(te_chunks) >= max_chunks_per_lang:
                break
        if len(te_chunks) >= max_chunks_per_lang:
            break

    print(f"[BuildMultilingual] Telugu chunks: {len(te_chunks)}", flush=True)

    chunks_map = {
        "en": en_chunks,
        "hi": hi_chunks,
        "te": te_chunks
    }
    return chunks_map, eval_queries

def main():
    start_time = time.time()
    max_c = int(os.getenv("MAX_CHUNKS_PER_LANGUAGE", "1500"))
    print("==================================================", flush=True)
    print(f"BUILDING MULTILINGUAL RETRIEVAL INDEXES ({max_c} chunks/language)", flush=True)
    print("==================================================", flush=True)

    chunks_map, eval_queries = build_multilingual_chunks(max_chunks_per_lang=max_c)
    embed_service = get_embedding_service()

    indexes_base_dir = DATA_DIR / "indexes"
    os.makedirs(indexes_base_dir, exist_ok=True)

    summary_stats = {}

    for lang in ["en", "hi", "te"]:
        lang_chunks = chunks_map[lang]
        print(f"\n--- Processing Language: {lang.upper()} ({len(lang_chunks)} chunks) ---", flush=True)
        lang_dir = indexes_base_dir / lang
        os.makedirs(lang_dir, exist_ok=True)

        processed_path = lang_dir / "processed_chunks.json"
        with open(processed_path, "w", encoding="utf-8") as f:
            json.dump(lang_chunks, f, ensure_ascii=False, indent=2)

        # Generate Embeddings
        texts = [c["text"] for c in lang_chunks]
        t0 = time.time()
        embeddings = embed_service.encode(texts, normalize=True, batch_size=256, show_progress=False)
        print(f"  [{lang.upper()}] Encoded {len(texts)} chunks (shape: {embeddings.shape}) in {time.time() - t0:.2f}s", flush=True)

        # Build FAISS
        faiss_store = FAISSVectorStore()
        faiss_store.index_path = str(lang_dir / "faiss_index.bin")
        faiss_store.metadata_path = str(lang_dir / "faiss_metadata.json")
        faiss_store.build_index(embeddings, lang_chunks)
        faiss_store.save()

        # Build BM25
        bm25_retriever = BM25Retriever()
        bm25_retriever.index_path = str(lang_dir / "bm25_index.pkl")
        bm25_retriever.build_index(lang_chunks)
        bm25_retriever.save()

        faiss_size_mb = os.path.getsize(lang_dir / "faiss_index.bin") / (1024 * 1024)
        bm25_size_mb = os.path.getsize(lang_dir / "bm25_index.pkl") / (1024 * 1024)
        
        sample_target_lang = lang_chunks[0]["target_lang"] if lang_chunks else "unknown"

        summary_stats[lang] = {
            "chunks_count": len(lang_chunks),
            "faiss_size_mb": round(faiss_size_mb, 2),
            "bm25_size_mb": round(bm25_size_mb, 2),
            "target_lang": sample_target_lang
        }

        # Root fallback copy for backward compatibility
        if lang == "hi":
            with open(settings.PROCESSED_CHUNKS_PATH, "w", encoding="utf-8") as f:
                json.dump(lang_chunks, f, ensure_ascii=False, indent=2)
            faiss_store.index_path = settings.FAISS_INDEX_PATH
            faiss_store.metadata_path = settings.FAISS_METADATA_PATH
            faiss_store.save()
            bm25_retriever.index_path = settings.BM25_INDEX_PATH
            bm25_retriever.save()

    total_time = time.time() - start_time
    print("\n==================================================", flush=True)
    print("MULTILINGUAL INDEX BUILDING COMPLETE", flush=True)
    print("==================================================", flush=True)
    for lang, stats in summary_stats.items():
        print(f"Language: {lang.upper()}", flush=True)
        print(f"  Chunks: {stats['chunks_count']}", flush=True)
        print(f"  FAISS Size: {stats['faiss_size_mb']} MB", flush=True)
        print(f"  BM25 Size:  {stats['bm25_size_mb']} MB", flush=True)
        print(f"  target_lang metadata: {stats['target_lang']}", flush=True)
    print(f"Total Time: {total_time:.2f}s", flush=True)
    print("==================================================", flush=True)

if __name__ == "__main__":
    main()
