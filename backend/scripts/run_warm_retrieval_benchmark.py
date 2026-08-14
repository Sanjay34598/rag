import sys
import io
import time
import json
from pathlib import Path

# Configure UTF-8 for stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings
from app.services.retrieval.retrieval_service import get_retrieval_service

def load_hindi_queries(limit=60):
    chunks_path = Path(settings.PROCESSED_CHUNKS_PATH)
    queries = []
    if chunks_path.exists():
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            for c in chunks:
                text = c.get("text", "").strip()
                if len(text) > 10:
                    q = text.split("।")[0].split(".")[0].strip()
                    if len(q) > 15 and q not in queries:
                        queries.append(q)
    
    if len(queries) < limit:
        base_queries = [
            "कॉर्पोरेशन क्या है?",
            "रेचल कार्सन ने क्यों एक दायित्व बर्दाश्त करने के लिए लिखा",
            "पोटेशियम में कम खाद्य पदार्थों का चार्ट।",
            "मालवाहक जहाज़ के नीचे की तरफ",
            "ईमानदारी या सच्चाई की परिभाषा",
            "लिंकन में अब वायुमंडलीय दबाव क्या है?",
            "स्ट्रूथर्स शहर स्कूल जिला राज्य संख्या",
            "क्या चिकित्सीय मारिजुआना मदत करता है?",
            "फ्रैंक गिफोर्ड ने कितनी महिलाओं से शादी की",
            "कितनी ट्रम्प प्रशासन की जांच चल रही हैं"
        ]
        while len(queries) < limit:
            queries.extend(base_queries)
            
    return queries[:limit]

def main():
    print("==================================================")
    print("CONTROLLED WARM RETRIEVAL BENCHMARK (50 WARM QUERIES)")
    print("==================================================")
    
    retrieval_service = get_retrieval_service()
    
    all_queries = load_hindi_queries(60)
    warmup_queries = all_queries[:10]
    test_queries = all_queries[10:60]
    
    print(f"Total queries loaded: {len(all_queries)}")
    print(f"Warmup queries      : {len(warmup_queries)} (Will be discarded)")
    print(f"Test queries        : {len(test_queries)}")
    
    # 1. Warmup Phase
    print("\n--- WARMUP PHASE ---")
    for i, q in enumerate(warmup_queries, start=1):
        _ = retrieval_service.retrieve(q, top_k=5)
    print("Warmup complete. Models & CPU caches primed.\n")
    
    # 2. Measurement Phase
    emb_times, faiss_times, bm25_times, fusion_times, total_times = [], [], [], [], []
    query_records = []
    
    for i, q in enumerate(test_queries, start=1):
        t0 = time.perf_counter()
        
        # 1. Embedding
        t_emb_start = time.perf_counter()
        query_emb = retrieval_service.embedding_service.encode_query(q)
        emb_ms = (time.perf_counter() - t_emb_start) * 1000.0
        
        # 2. FAISS
        t_faiss_start = time.perf_counter()
        dense_results = retrieval_service.vector_store.search(query_emb, top_k=settings.CANDIDATE_K)
        faiss_ms = (time.perf_counter() - t_faiss_start) * 1000.0
        
        # 3. BM25
        t_bm25_start = time.perf_counter()
        sparse_results = retrieval_service.bm25_retriever.search(q, top_k=settings.CANDIDATE_K)
        bm25_ms = (time.perf_counter() - t_bm25_start) * 1000.0
        
        # 4. Fusion
        t_fusion_start = time.perf_counter()
        fused_results = retrieval_service.hybrid_retriever.fuse(dense_results, sparse_results, top_k=settings.TOP_K)
        fusion_ms = (time.perf_counter() - t_fusion_start) * 1000.0
        
        tot_ms = (time.perf_counter() - t0) * 1000.0
        
        emb_times.append(emb_ms)
        faiss_times.append(faiss_ms)
        bm25_times.append(bm25_ms)
        fusion_times.append(fusion_ms)
        total_times.append(tot_ms)
        
        query_records.append({
            "query": q,
            "emb_ms": emb_ms,
            "faiss_ms": faiss_ms,
            "bm25_ms": bm25_ms,
            "fusion_ms": fusion_ms,
            "total_ms": tot_ms
        })
        
    def stats(arr):
        s = sorted(arr)
        n = len(s)
        avg = sum(s) / n
        p50 = s[int(0.50 * (n - 1))]
        p70 = s[int(0.70 * (n - 1))]
        p90 = s[int(0.90 * (n - 1))]
        p95 = s[int(0.95 * (n - 1))]
        p99 = s[int(0.99 * (n - 1))]
        p100 = s[-1]
        return avg, p50, p70, p90, p95, p99, p100

    print("--- BENCHMARK RESULTS (50 WARM QUERIES) ---")
    emb_avg, emb_p50, emb_p70, emb_p90, emb_p95, emb_p99, emb_p100 = stats(emb_times)
    faiss_avg, faiss_p50, faiss_p70, faiss_p90, faiss_p95, faiss_p99, faiss_p100 = stats(faiss_times)
    bm25_avg, bm25_p50, bm25_p70, bm25_p90, bm25_p95, bm25_p99, bm25_p100 = stats(bm25_times)
    tot_avg, tot_p50, tot_p70, tot_p90, tot_p95, tot_p99, tot_p100 = stats(total_times)
    
    print(f"Embedding Latency: Avg={emb_avg:.2f}ms | P50={emb_p50:.2f}ms | P70={emb_p70:.2f}ms | P90={emb_p90:.2f}ms | P95={emb_p95:.2f}ms | P99={emb_p99:.2f}ms | P100={emb_p100:.2f}ms")
    print(f"FAISS Latency    : Avg={faiss_avg:.2f}ms | P50={faiss_p50:.2f}ms | P70={faiss_p70:.2f}ms | P90={faiss_p90:.2f}ms | P95={faiss_p95:.2f}ms | P99={faiss_p99:.2f}ms | P100={faiss_p100:.2f}ms")
    print(f"BM25 Latency     : Avg={bm25_avg:.2f}ms | P50={bm25_avg:.2f}ms | P70={bm25_p70:.2f}ms | P90={bm25_p90:.2f}ms | P95={bm25_p95:.2f}ms | P99={bm25_p99:.2f}ms | P100={bm25_p100:.2f}ms")
    print(f"TOTAL RETRIEVAL  : Avg={tot_avg:.2f}ms | P50={tot_p50:.2f}ms | P70={tot_p70:.2f}ms | P90={tot_p90:.2f}ms | P95={tot_p95:.2f}ms | P99={tot_p99:.2f}ms | P100={tot_p100:.2f}ms")

    # Slowest 5 queries
    sorted_by_total = sorted(query_records, key=lambda x: x["total_ms"], reverse=True)
    print("\n--- SLOWEST 5 WARM QUERIES ---")
    for i, rec in enumerate(sorted_by_total[:5], start=1):
        print(f"#{i}: Total={rec['total_ms']:.2f}ms (Emb={rec['emb_ms']:.2f}ms, BM25={rec['bm25_ms']:.2f}ms) | Query: '{rec['query'][:40]}...'")

if __name__ == "__main__":
    main()
