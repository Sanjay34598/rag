import os
import sys
import json
import time

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings, DATA_DIR
from app.services.rag.rag_service import get_rag_service

def main():
    print("==================================================")
    print("STAGE 4: END-TO-END RAG PIPELINE VERIFICATION")
    print("==================================================")

    eval_queries_path = os.path.join(DATA_DIR, "eval_queries.json")
    if not os.path.exists(eval_queries_path):
        raise FileNotFoundError("eval_queries.json missing. Run build_indexes.py first.")

    with open(eval_queries_path, "r", encoding="utf-8") as f:
        eval_queries = json.load(f)

    # Select 10 real queries from the MSMARCO-XI dataset
    sample_queries = [q["query"] for q in eval_queries[:10] if q.get("query")]

    print(f"[RAGTest] Testing {len(sample_queries)} real queries through the full RAG harness...\n")

    rag = get_rag_service()
    rag.initialize(load_indexes=True)

    results_log = []
    total_latencies = []

    for idx, query in enumerate(sample_queries, start=1):
        print(f"--- Query {idx}/{len(sample_queries)} ---")
        print(f"User Query: {query}")
        
        response = rag.answer(query)
        
        print(f"Answer:     {response['answer']}")
        print(f"Grounded:   {response['grounded']} (Confidence: {response['confidence']})")
        print(f"Sources:    {len(response['sources'])} retrieved source chunks")
        lat = response["latency"]
        print(f"Latency:    Retrieval: {lat['retrieval_ms']}ms | LLM: {lat['llm_ms']}ms | Total: {lat['total_ms']}ms\n")
        
        results_log.append(response)
        total_latencies.append(lat["total_ms"])

    avg_latency = sum(total_latencies) / len(total_latencies)
    grounded_count = sum(1 for r in results_log if r["grounded"])

    print("==================================================")
    print("STAGE 4 VERIFICATION SUMMARY")
    print("==================================================")
    print(f"Total Queries Executed:  {len(sample_queries)}")
    print(f"Successfully Answered:   {len(results_log)}")
    print(f"Grounded Answers:        {grounded_count}/{len(sample_queries)}")
    print(f"Average Pipeline Latency:{avg_latency:.2f} ms")
    print("==================================================")

    # Save test log
    log_path = os.path.join(DATA_DIR, "rag_verification_results.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(results_log, f, ensure_ascii=False, indent=2)
    print(f"[RAGTest] Results saved to {log_path}")

if __name__ == "__main__":
    main()
