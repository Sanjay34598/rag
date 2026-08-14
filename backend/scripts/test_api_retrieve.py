import os
import sys
import json
from fastapi.testclient import TestClient

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.core.config import DATA_DIR

def test_api():
    client = TestClient(app)
    
    eval_queries_path = os.path.join(DATA_DIR, "eval_queries.json")
    with open(eval_queries_path, "r", encoding="utf-8") as f:
        eval_queries = json.load(f)

    sample_queries = [q["query"] for q in eval_queries[:10] if q.get("query")]

    print("==================================================")
    print("TESTING POST /api/v1/retrieve (RERANKER_ENABLED=false)")
    print("==================================================")

    for idx, q in enumerate(sample_queries, start=1):
        response = client.post("/api/v1/retrieve", json={"query": q, "top_k": 5})
        assert response.status_code == 200, f"Error {response.status_code}: {response.text}"
        data = response.json()
        
        print(f"Query {idx}: '{q}'")
        print(f"  Status:             {response.status_code}")
        print(f"  reranking_enabled:  {data['reranking_enabled']}")
        print(f"  latency_ms:         {data['latency_ms']} ms")
        print(f"  reranking_ms:       {data['latency_breakdown']['reranking_ms']} ms")
        print(f"  Results Count:      {len(data['results'])}\n")

        assert data["reranking_enabled"] == False
        assert data["latency_breakdown"]["reranking_ms"] == 0.0
        assert len(data["results"]) <= 5

    print("API verification completed successfully!")

if __name__ == "__main__":
    test_api()
