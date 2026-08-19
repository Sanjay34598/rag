import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.main import app
from app.services.rag.rag_service import get_rag_service

service = get_rag_service()
service.initialize(load_indexes=False)

client = TestClient(app)

print("1. Health Endpoint:", flush=True)
h_res = client.get("/health")
print("Status:", h_res.status_code, h_res.json(), flush=True)

print("\n2. Retrieve Endpoint (Lazy Loading Test):", flush=True)
ret_res = client.post("/api/v1/retrieve", json={"query": "What is a corporation?"})
print("Status:", ret_res.status_code, flush=True)
if ret_res.status_code == 200:
    data = ret_res.json()
    print("Results count:", len(data.get("results", [])), flush=True)
    print("Latency breakdown:", data.get("latency_breakdown"), flush=True)

print("\n3. RAG Query Endpoint:", flush=True)
rag_res = client.post("/api/v1/rag/query", json={"query": "What is a corporation?", "language_code": "en-IN"})
print("Status:", rag_res.status_code, flush=True)
if rag_res.status_code == 200:
    data = rag_res.json()
    print("Grounded:", data.get("grounded"), flush=True)
    print("Confidence:", data.get("confidence"), flush=True)
    print("Answer:", data.get("answer")[:150], flush=True)
