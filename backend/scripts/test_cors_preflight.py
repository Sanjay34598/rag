import os
import sys
from fastapi.testclient import TestClient

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app

def run_cors_tests():
    client = TestClient(app)

    print("==================================================")
    print("VERIFYING BACKEND CORS PREFLIGHT & POST ENDPOINTS")
    print("==================================================")

    test_origins = [
        "https://voice-che0n6xg2-sanjays-projects-f2a71297.vercel.app",
        "https://voice-rag-alpha.vercel.app"
    ]

    all_passed = True

    for origin in test_origins:
        print(f"\n--- Testing OPTIONS Preflight for Origin: {origin} ---")
        headers = {
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type"
        }
        res = client.options("/api/v1/rag/query", headers=headers)
        
        status_ok = res.status_code in (200, 204)
        allowed_orig = res.headers.get("access-control-allow-origin") == origin
        allowed_methods = res.headers.get("access-control-allow-methods") or "*"
        
        print(f"  Response Status  : {res.status_code}")
        print(f"  Allow-Origin     : {res.headers.get('access-control-allow-origin')}")
        print(f"  Allow-Credentials: {res.headers.get('access-control-allow-credentials')}")
        print(f"  Allow-Methods    : {res.headers.get('access-control-allow-methods')}")

        if status_ok and allowed_orig:
            print("  Status           : [PASS]")
        else:
            print("  Status           : [FAIL]")
            all_passed = False

    print("\n--- Testing POST /api/v1/rag/query with CORS Origin ---")
    from unittest.mock import patch
    from app.services.rag.rag_service import get_rag_service
    service = get_rag_service()
    service.is_ready = True

    post_origin = "https://voice-che0n6xg2-sanjays-projects-f2a71297.vercel.app"
    payload = {"query": "What is vitamin B2?", "language_code": "en-IN"}
    
    mock_rag_response = {
        "query": "What is vitamin B2?",
        "answer": "Vitamin B-2 is also known as Riboflavin.",
        "grounded": True,
        "confidence": 0.90,
        "sources": [{"chunk_id": "p_1090320_4", "text": "Vitamin B-2 (Riboflavin) 100 mg.", "score": 0.95}],
        "language_code": "en-IN",
        "latency": {
            "retrieval_ms": 2.0,
            "context_ms": 1.0,
            "llm_ms": 5.0,
            "grounding_ms": 1.0,
            "total_ms": 9.0
        }
    }
    
    with patch.object(service, "answer", return_value=mock_rag_response):
        res_post = client.post("/api/v1/rag/query", json=payload, headers={"Origin": post_origin})
    
    post_status_ok = res_post.status_code == 200
    post_cors_ok = res_post.headers.get("access-control-allow-origin") == post_origin
    ans = res_post.json().get("answer", "") if res_post.status_code == 200 else str(res_post.json())
    
    print(f"  POST Status Code : {res_post.status_code}")
    print(f"  Allow-Origin     : {res_post.headers.get('access-control-allow-origin')}")
    print(f"  Answer Output    : {ans[:60]}...")
    
    if post_status_ok and post_cors_ok:
        print("  Status           : [PASS]")
    else:
        print("  Status           : [FAIL]")
        all_passed = False

    print("\n==================================================")
    if all_passed:
        print("ALL CORS PREFLIGHT & POST TESTS PASSED!")
    else:
        print("CORS TESTS FAILED!")
    print("==================================================")

    return all_passed

if __name__ == "__main__":
    success = run_cors_tests()
    sys.exit(0 if success else 1)
