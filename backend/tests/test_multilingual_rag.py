import pytest
from app.services.rag.rag_service import get_rag_service, RAGService

@pytest.fixture(scope="module")
def rag_service():
    service = get_rag_service()
    service.initialize(load_indexes=True)
    return service

def test_1_english_query_returns_english_sources(rag_service: RAGService):
    res = rag_service.answer("What is a corporation?", language_code="en-IN")
    assert res["grounded"] is True
    assert res["language_code"] in ("en-IN", "en")
    assert len(res["sources"]) > 0
    for s in res["sources"]:
        assert s["language"] == "en", f"Expected source language 'en' but got '{s['language']}'"

def test_2_hindi_query_returns_hindi_sources(rag_service: RAGService):
    res = rag_service.answer("कॉर्पोरेशन क्या है?", language_code="hi-IN")
    assert res["grounded"] is True
    assert res["language_code"] in ("hi-IN", "hi")
    assert len(res["sources"]) > 0
    for s in res["sources"]:
        assert s["language"] == "hi", f"Expected source language 'hi' but got '{s['language']}'"

def test_3_telugu_query_returns_telugu_sources(rag_service: RAGService):
    res = rag_service.answer("కార్పొరేషన్ అంటే ఏమిటి?", language_code="te-IN")
    assert res["grounded"] is True
    assert res["language_code"] in ("te-IN", "te")
    assert len(res["sources"]) > 0
    for s in res["sources"]:
        assert s["language"] == "te", f"Expected source language 'te' but got '{s['language']}'"

def test_4_autodetect_english_query(rag_service: RAGService):
    res = rag_service.answer("What is a corporation?", language_code="Auto Detect")
    assert res["grounded"] is True
    assert res["language_code"] in ("en-IN", "en")
    for s in res["sources"]:
        assert s["language"] == "en"

def test_5_autodetect_hindi_query(rag_service: RAGService):
    res = rag_service.answer("कॉर्पोरेशन क्या है?", language_code="Auto Detect")
    assert res["grounded"] is True
    assert res["language_code"] in ("hi-IN", "hi")
    for s in res["sources"]:
        assert s["language"] == "hi"

def test_6_autodetect_telugu_query(rag_service: RAGService):
    res = rag_service.answer("కార్పొరేషన్ అంటే ఏమిటి?", language_code="Auto Detect")
    assert res["grounded"] is True
    assert res["language_code"] in ("te-IN", "te")
    for s in res["sources"]:
        assert s["language"] == "te"

def test_7_conversational_hello_returns_no_sources(rag_service: RAGService):
    res = rag_service.answer("hello", language_code="en-IN")
    assert res["grounded"] is False
    assert res["sources"] == []

def test_8_english_query_never_returns_hindi_sources(rag_service: RAGService):
    res = rag_service.answer("What is a corporation?", language_code="en-IN")
    for s in res["sources"]:
        assert s["language"] != "hi"
        assert s["language"] != "te"

def test_9_telugu_query_never_returns_hindi_sources(rag_service: RAGService):
    res = rag_service.answer("కార్పొరేషన్ అంటే ఏమిటి?", language_code="te-IN")
    for s in res["sources"]:
        assert s["language"] != "hi"
        assert s["language"] != "en"

def test_10_hindi_query_never_returns_telugu_sources(rag_service: RAGService):
    res = rag_service.answer("कॉर्पोरेशन क्या है?", language_code="hi-IN")
    for s in res["sources"]:
        assert s["language"] != "te"
        assert s["language"] != "en"
