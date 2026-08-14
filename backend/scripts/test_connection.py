import requests

def test_endpoints():
    print("==================================================")
    print("TESTING BACKEND DIRECT AND VITE PROXY ENDPOINTS")
    print("==================================================")
    
    # 1. Health check
    try:
        r = requests.get("http://127.0.0.1:8000/health", timeout=5)
        print("GET http://127.0.0.1:8000/health ->", r.status_code, r.text)
    except Exception as e:
        print("GET http://127.0.0.1:8000/health FAILED:", e)

    # 2. Direct Backend Text RAG
    try:
        r = requests.post("http://127.0.0.1:8000/api/v1/rag/query", json={"query": "कॉर्पोरेशन क्या है?"}, timeout=15)
        print("\nPOST http://127.0.0.1:8000/api/v1/rag/query ->", r.status_code)
        print("Response Body Snippet:", r.text[:300])
    except Exception as e:
        print("\nPOST http://127.0.0.1:8000/api/v1/rag/query FAILED:", e)

    # 3. Vite Proxy Text RAG (port 3000)
    try:
        r = requests.post("http://localhost:3000/api/v1/rag/query", json={"query": "कॉर्पोरेशन क्या है?"}, timeout=15)
        print("\nPOST http://localhost:3000/api/v1/rag/query (Vite Proxy) ->", r.status_code)
        print("Response Body Snippet:", r.text[:300])
    except Exception as e:
        print("\nPOST http://localhost:3000/api/v1/rag/query FAILED:", e)

if __name__ == "__main__":
    test_endpoints()
