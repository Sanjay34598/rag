import pytest
from unittest.mock import patch
from app.services.rag.rag_service import get_rag_service
from app.core.config import settings

@pytest.fixture(scope="module")
def rag_service():
    service = get_rag_service()
    service.initialize(load_indexes=True)
    return service

def test_1_what_is_a_corporation(rag_service, monkeypatch):
    monkeypatch.setattr(settings, "LLM_MODE", "fallback")
    monkeypatch.setattr(rag_service.answer_generator, "api_key", "")
    res = rag_service.answer("What is a corporation?", language_code="en-IN")
    assert res["grounded"] is True
    assert res["confidence"] > 0.0
    assert len(res["sources"]) > 0

def test_2_company_incorporated_query(rag_service, monkeypatch):
    monkeypatch.setattr(settings, "LLM_MODE", "fallback")
    monkeypatch.setattr(rag_service.answer_generator, "api_key", "")
    res = rag_service.answer("A company is incorporated in a specific nation.", language_code="en-IN")
    assert res["grounded"] is True
    assert res["confidence"] > 0.0
    assert len(res["sources"]) > 0

def test_3_unrelated_question_refused(rag_service, monkeypatch):
    monkeypatch.setattr(settings, "LLM_MODE", "fallback")
    monkeypatch.setattr(rag_service.answer_generator, "api_key", "")
    res = rag_service.answer("Who won the Mars Rovers Race in 2099?", language_code="en-IN")
    assert res["grounded"] is False
    assert res["confidence"] == 0.0
    assert res["sources"] == []

def test_4_hindi_corporation_query(rag_service, monkeypatch):
    monkeypatch.setattr(settings, "LLM_MODE", "fallback")
    monkeypatch.setattr(rag_service.answer_generator, "api_key", "")
    res = rag_service.answer("कॉर्पोरेशन क्या है?", language_code="hi-IN")
    assert res["grounded"] is True
    assert res["confidence"] > 0.0
    assert len(res["sources"]) > 0

def test_5_telugu_corporation_query(rag_service, monkeypatch):
    monkeypatch.setattr(settings, "LLM_MODE", "fallback")
    monkeypatch.setattr(rag_service.answer_generator, "api_key", "")
    res = rag_service.answer("కార్పొరేషన్ అంటే ఏమిటి?", language_code="te-IN")
    assert res["grounded"] is True
    assert res["confidence"] > 0.0
    assert len(res["sources"]) > 0

from app.services.rag.answer_generator import AnswerGenerator

@patch.object(AnswerGenerator, "_call_groq_api")
def test_6_real_llm_mode_incorporated_query(mock_groq, rag_service, monkeypatch):
    monkeypatch.setattr(settings, "LLM_MODE", "real")
    monkeypatch.setattr(rag_service.answer_generator, "api_key", "dummy_key")
    mock_groq.return_value = '{"answer": "A company or corporation is incorporated in a specific nation under the authority of law to conduct business.", "grounded": true, "confidence": 0.95}'
    
    res = rag_service.answer("A company is incorporated in a specific nation.", language_code="en-IN")
    assert res["grounded"] is True
    assert res["confidence"] > 0.0
    assert len(res["sources"]) > 0
    assert "incorporated" in res["answer"].lower()
