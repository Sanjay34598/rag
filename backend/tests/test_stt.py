import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings

client = TestClient(app)

def test_transcribe_missing_api_key(monkeypatch):
    monkeypatch.setattr(settings, "SARVAM_API_KEY", "")
    
    response = client.post(
        "/api/v1/stt/transcribe",
        files={"file": ("test.wav", b"dummy audio content", "audio/wav")}
    )
    assert response.status_code == 503
    assert "SARVAM_API_KEY is missing" in response.json()["detail"]

def test_transcribe_empty_audio():
    response = client.post(
        "/api/v1/stt/transcribe",
        files={"file": ("test.wav", b"", "audio/wav")}
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()

def test_transcribe_unsupported_audio_format():
    response = client.post(
        "/api/v1/stt/transcribe",
        files={"file": ("test.txt", b"some text file content", "text/plain")}
    )
    assert response.status_code == 415
    assert "unsupported" in response.json()["detail"].lower()

@patch("requests.post")
def test_transcribe_success(mock_post, monkeypatch):
    monkeypatch.setattr(settings, "SARVAM_API_KEY", "dummy_sarvam_key")
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "transcript": "कॉर्पोरेशन क्या है?",
        "language_code": "hi-IN"
    }
    mock_post.return_value = mock_resp

    response = client.post(
        "/api/v1/stt/transcribe",
        files={"file": ("test.wav", b"fake audio content bytes", "audio/wav")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["transcript"] == "कॉर्पोरेशन क्या है?"
    assert data["language"] == "hi-IN"
    assert data["confidence"] == 1.0
    assert "latency_ms" in data

@patch("requests.post")
def test_transcribe_empty_transcript(mock_post, monkeypatch):
    monkeypatch.setattr(settings, "SARVAM_API_KEY", "dummy_sarvam_key")
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "transcript": "   ",
        "language_code": "hi-IN"
    }
    mock_post.return_value = mock_resp

    response = client.post(
        "/api/v1/stt/transcribe",
        files={"file": ("test.wav", b"fake audio content bytes", "audio/wav")}
    )
    assert response.status_code == 422
    assert "Empty transcription" in response.json()["detail"]
