import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings

client = TestClient(app)

@patch("requests.post")
def test_voice_query_success(mock_post, monkeypatch):
    monkeypatch.setattr(settings, "SARVAM_API_KEY", "dummy_sarvam_key")
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "transcript": "कॉर्पोरेशन क्या है?",
        "language_code": "hi-IN"
    }
    mock_post.return_value = mock_resp

    response = client.post(
        "/api/v1/voice/query",
        files={"file": ("sample.wav", b"fake audio data content", "audio/wav")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["transcript"] == "कॉर्पोरेशन क्या है?"
    assert "answer" in data
    assert "grounded" in data
    assert "sources" in data
    assert "latency" in data
    assert "stt_ms" in data["latency"]
    assert "retrieval_ms" in data["latency"]
    assert "total_ms" in data["latency"]

def test_voice_query_missing_api_key(monkeypatch):
    monkeypatch.setattr(settings, "SARVAM_API_KEY", "")
    
    response = client.post(
        "/api/v1/voice/query",
        files={"file": ("sample.wav", b"fake audio data content", "audio/wav")}
    )
    assert response.status_code == 503
    assert "Voice Query unavailable" in response.json()["detail"]
