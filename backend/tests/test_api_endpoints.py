import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"
    assert "version" in data
    assert "rag_ready" in data

def test_cors_test_endpoint():
    response = client.get("/cors-test")
    assert response.status_code == 200
    data = response.json()
    assert data == {"status": "ok", "message": "CORS is working"}

def test_cors_headers_allowed_vercel_origin():
    target_origin = "https://voice-b0064qrq6-sanjays-projects-f2a71297.vercel.app"
    response = client.options(
        "/cors-test",
        headers={
            "Origin": target_origin,
            "Access-Control-Request-Method": "GET"
        }
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == target_origin

def test_cors_headers_allowed_localhost_origin():
    target_origin = "http://localhost:5173"
    response = client.options(
        "/cors-test",
        headers={
            "Origin": target_origin,
            "Access-Control-Request-Method": "GET"
        }
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == target_origin
