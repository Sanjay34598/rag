import pytest
from unittest.mock import patch
from app.services.rag.rag_service import get_rag_service
from app.services.rag.answer_generator import AnswerGenerator
from app.core.config import settings

@pytest.fixture(scope="module")
def rag_service():
    service = get_rag_service()
    service.initialize(load_indexes=True)
    return service

def test_corporation_english_query_fallback(rag_service, monkeypatch):
    monkeypatch.setattr(settings, "LLM_MODE", "fallback")
    monkeypatch.setattr(rag_service.answer_generator, "api_key", "")
    res = rag_service.answer("What is a corporation?", language_code="en-IN")
    assert res["grounded"] is True
    assert res["confidence"] > 0.0
    assert len(res["sources"]) > 0
    assert any("corporation" in s["text"].lower() for s in res["sources"])

def test_corporation_hindi_query_fallback(rag_service, monkeypatch):
    monkeypatch.setattr(settings, "LLM_MODE", "fallback")
    monkeypatch.setattr(rag_service.answer_generator, "api_key", "")
    res = rag_service.answer("कॉर्पोरेशन क्या है?", language_code="hi-IN")
    assert res["grounded"] is True
    assert res["confidence"] > 0.0
    assert len(res["sources"]) > 0
    assert any("corporation" in s["text"].lower() for s in res["sources"])

def test_corporation_telugu_query_fallback(rag_service, monkeypatch):
    monkeypatch.setattr(settings, "LLM_MODE", "fallback")
    monkeypatch.setattr(rag_service.answer_generator, "api_key", "")
    res = rag_service.answer("కార్పొరేషన్ అంటే ఏమిటి?", language_code="te-IN")
    assert res["grounded"] is True
    assert res["confidence"] > 0.0
    assert len(res["sources"]) > 0
    assert any("corporation" in s["text"].lower() for s in res["sources"])

@patch.object(AnswerGenerator, "_call_groq_api")
def test_corporation_mocked_llm_response(mock_groq, rag_service, monkeypatch):
    monkeypatch.setattr(settings, "LLM_MODE", "real")
    monkeypatch.setattr(rag_service.answer_generator, "api_key", "dummy_key")
    mock_groq.return_value = '{"answer": "A corporation is an association of individuals created by law having a continuous existence.", "grounded": true, "confidence": 0.95}'
    
    res = rag_service.answer("What is a corporation?", language_code="en-IN")
    assert res["grounded"] is True
    assert res["confidence"] > 0.0
    assert len(res["sources"]) > 0
    assert "corporation" in res["answer"].lower()



