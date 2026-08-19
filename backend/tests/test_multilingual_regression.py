import pytest
import re
from unittest.mock import patch
from app.services.rag.rag_service import get_rag_service
from app.services.rag.answer_generator import AnswerGenerator
from app.services.rag.grounding_validator import validate_language_output
from app.core.config import settings

@pytest.fixture(scope="module")
def rag_service():
    service = get_rag_service()
    service.initialize(load_indexes=True)
    return service

def test_A_english_corporation_query(rag_service, monkeypatch):
    """Test A: English query with en-IN should return English answer, grounded=True, sources non-empty."""
    monkeypatch.setattr(settings, "LLM_MODE", "fallback")
    monkeypatch.setattr(rag_service.answer_generator, "api_key", "")
    res = rag_service.answer("What is a corporation?", language_code="en-IN")
    assert res["grounded"] is True
    assert res["confidence"] > 0.0
    assert len(res["sources"]) > 0
    assert "couldn't verify" not in res["answer"].lower()

def test_B_hindi_query_hi_IN(rag_service, monkeypatch):
    """Test B: Hindi query with hi-IN should return Hindi/Devanagari answer, NOT refusal, grounded=True, sources non-empty."""
    monkeypatch.setattr(settings, "LLM_MODE", "fallback")
    monkeypatch.setattr(rag_service.answer_generator, "api_key", "")
    res = rag_service.answer("कंपनी क्या है?", language_code="hi-IN")
    assert res["grounded"] is True
    assert res["confidence"] > 0.0
    assert len(res["sources"]) > 0
    assert validate_language_output(res["answer"], "hi-IN") is True
    assert "पुष्टि नहीं मिली" not in res["answer"]

def test_C_english_query_hi_IN(rag_service, monkeypatch):
    """Test C: English query with hi-IN should return Hindi/Devanagari answer, grounded=True, sources non-empty."""
    monkeypatch.setattr(settings, "LLM_MODE", "fallback")
    monkeypatch.setattr(rag_service.answer_generator, "api_key", "")
    res = rag_service.answer("What is a corporation?", language_code="hi-IN")
    assert res["grounded"] is True
    assert res["confidence"] > 0.0
    assert len(res["sources"]) > 0
    assert validate_language_output(res["answer"], "hi-IN") is True

def test_D_telugu_query_te_IN(rag_service, monkeypatch):
    """Test D: Telugu query with te-IN should return Telugu answer, grounded=True, sources non-empty."""
    monkeypatch.setattr(settings, "LLM_MODE", "fallback")
    monkeypatch.setattr(rag_service.answer_generator, "api_key", "")
    res = rag_service.answer("కార్పొరేషన్ అంటే ఏమిటి?", language_code="te-IN")
    assert res["grounded"] is True
    assert res["confidence"] > 0.0
    assert len(res["sources"]) > 0
    assert validate_language_output(res["answer"], "te-IN") is True

def test_E_unrelated_query_refusal(rag_service, monkeypatch):
    """Test E: Clearly unrelated question should return grounded=False / safe refusal."""
    monkeypatch.setattr(settings, "LLM_MODE", "fallback")
    monkeypatch.setattr(rag_service.answer_generator, "api_key", "")
    res = rag_service.answer("Who won the Mars Rovers Race in 2099?", language_code="en-IN")
    assert res["grounded"] is False
    assert res["confidence"] == 0.0
    assert res["sources"] == []

@patch.object(AnswerGenerator, "_call_groq_api")
def test_real_llm_mode_hindi_translation(mock_groq, rag_service, monkeypatch):
    """Verify Groq LLM real mode returning Hindi answer passes validation."""
    monkeypatch.setattr(settings, "LLM_MODE", "real")
    monkeypatch.setattr(rag_service.answer_generator, "api_key", "dummy_key")
    mock_groq.return_value = '{"answer": "एक निगम कानून द्वारा स्थापित व्यक्तियों का एक संघ है जिसकी एक निरंतर उपस्थिति होती है।", "grounded": true, "confidence": 0.95}'
    
    res = rag_service.answer("What is a corporation?", language_code="hi-IN")
    assert res["grounded"] is True
    assert res["confidence"] > 0.0
    assert len(res["sources"]) > 0
    assert validate_language_output(res["answer"], "hi-IN") is True
