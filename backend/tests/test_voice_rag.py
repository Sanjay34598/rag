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

@patch("requests.post")
def test_voice_query_api_error_handling(mock_post, monkeypatch):
    monkeypatch.setattr(settings, "SARVAM_API_KEY", "dummy_sarvam_key")
    
    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.text = "Bad Request from Sarvam"
    mock_post.return_value = mock_resp

    response = client.post(
        "/api/v1/voice/query",
        files={"file": ("sample.wav", b"fake audio data content", "audio/wav")}
    )
    assert response.status_code == 400
    assert response.status_code != 500

@patch("app.services.stt.sarvam_stt.SarvamSTTService.transcribe")
def test_voice_query_unrelated_courtesy_refusal(mock_transcribe, monkeypatch):
    monkeypatch.setattr(settings, "SARVAM_API_KEY", "dummy_sarvam_key")
    
    for courtesy in ["आपके लिए धन्यवाद", "धन्यवाद", "नमस्ते", "अलविदा"]:
        mock_transcribe.return_value = (True, {"transcript": courtesy, "language_code": "hi-IN"}, 45.0)

        response = client.post(
            "/api/v1/voice/query",
            files={"file": ("sample.wav", b"fake audio data content", "audio/wav")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["transcript"] == courtesy
        assert data["grounded"] is False
        assert data["confidence"] == 0.0
        assert data["sources"] == []
        assert "एशली" not in data["answer"]
        assert "मीथेन" not in data["answer"]

@patch("app.services.stt.sarvam_stt.SarvamSTTService.transcribe")
def test_voice_query_exact_transcript_propagation(mock_transcribe, monkeypatch):
    monkeypatch.setattr(settings, "SARVAM_API_KEY", "dummy_sarvam_key")
    
    exact_transcript = "वायुमंडलीय दबाव की परिभाषा"
    mock_transcribe.return_value = (True, {"transcript": exact_transcript, "language_code": "hi-IN"}, 50.0)

    response = client.post(
        "/api/v1/voice/query",
        files={"file": ("sample.wav", b"fake audio data content", "audio/wav")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["transcript"] == exact_transcript
    assert "answer" in data
    assert "sources" in data

@patch("app.services.stt.sarvam_stt.SarvamSTTService.transcribe")
def test_voice_query_courtesy_term_meaning_knowledge_query(mock_transcribe, monkeypatch):
    monkeypatch.setattr(settings, "SARVAM_API_KEY", "dummy_sarvam_key")
    
    for term_query in ["धन्यवाद शब्द का अर्थ क्या है?", "नमस्ते शब्द का अर्थ क्या है?"]:
        mock_transcribe.return_value = (True, {"transcript": term_query, "language_code": "hi-IN"}, 50.0)

        response = client.post(
            "/api/v1/voice/query",
            files={"file": ("sample.wav", b"fake audio data content", "audio/wav")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["transcript"] == term_query
        assert "latency" in data
        assert data["latency"]["retrieval_ms"] >= 0.0

