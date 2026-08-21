import os
import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.rag.rag_service import get_rag_service

def main():
    print("==================================================")
    print("RUNNING VITAMIN RETRIEVAL & GROUNDING TEST SUITE")
    print("==================================================")

    rag_service = get_rag_service()
    rag_service.initialize(load_indexes=True)

    queries = [
        "What is vitamin B?",
        "What is vitamin B12?",
        "What is vitamin D?",
        "What is vitamin B2?",
        "What are the benefits of vitamin B2?",
        "What is riboflavin?",
        "What are the benefits of riboflavin?"
    ]

    results = []

    for idx, q in enumerate(queries, 1):
        print(f"\n--------------------------------------------------")
        print(f"TEST {idx}: '{q}'")
        print(f"--------------------------------------------------")
        res = rag_service.answer(q)
        grounded = res["grounded"]
        sources = [s["chunk_id"] for s in res.get("sources", [])]
        answer_snippet = res["answer"][:100].replace("\n", " ")

        print(f"Result for '{q}':")
        print(f"  Grounded: {grounded}")
        print(f"  Confidence: {res.get('confidence', 0.0)}")
        print(f"  Sources: {sources}")
        print(f"  Answer: '{answer_snippet}...'")

        results.append({
            "query": q,
            "grounded": grounded,
            "sources": sources,
            "answer": answer_snippet
        })

    print("\n==================================================")
    print("SUMMARY OF TEST RESULTS")
    print("==================================================")
    print(f"{'Query':<40} | {'Grounded':<10} | {'Selected Source Chunks'}")
    print("-" * 80)
    for r in results:
        srcs_str = ", ".join(r["sources"]) if r["sources"] else "[]"
        print(f"{r['query']:<40} | {str(r['grounded']):<10} | {srcs_str}")
    print("==================================================")

if __name__ == "__main__":
    main()
