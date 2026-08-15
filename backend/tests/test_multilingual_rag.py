import pytest
from app.services.rag.rag_service import get_rag_service, RAGService

@pytest.fixture(scope="module")
def rag_service():
    service = get_rag_service()
    service.initialize(load_indexes=True)
    return service

def test_1_english_query_returns_canonical_evidence(rag_service: RAGService):
    res = rag_service.answer("What is a corporation?", language_code="en-IN")
    assert res["language_code"] in ("en-IN", "en")
    if res.get("sources"):
        for s in res["sources"]:
            assert s["language"] == "en", f"Expected canonical source language 'en' but got '{s['language']}'"

def test_2_hindi_query_returns_canonical_evidence_with_hindi_response(rag_service: RAGService):
    res = rag_service.answer("कॉर्पोरेशन क्या है?", language_code="hi-IN")
    assert res["language_code"] in ("hi-IN", "hi")
    if res.get("sources"):
        for s in res["sources"]:
            assert s["language"] == "en", f"Expected canonical evidence 'en' but got '{s['language']}'"

def test_3_telugu_query_returns_canonical_evidence_with_telugu_response(rag_service: RAGService):
    res = rag_service.answer("కార్పొరేషన్ అంటే ఏమిటి?", language_code="te-IN")
    assert res["language_code"] in ("te-IN", "te")
    if res.get("sources"):
        for s in res["sources"]:
            assert s["language"] == "en", f"Expected canonical evidence 'en' but got '{s['language']}'"

def test_4_autodetect_english_query(rag_service: RAGService):
    res = rag_service.answer("What is a corporation?", language_code="Auto Detect")
    assert res["language_code"] in ("en-IN", "en")
    if res.get("sources"):
        for s in res["sources"]:
            assert s["language"] == "en"

def test_5_autodetect_hindi_query(rag_service: RAGService):
    res = rag_service.answer("कॉर्पोरेशन क्या है?", language_code="Auto Detect")
    assert res["language_code"] in ("hi-IN", "hi")
    if res.get("sources"):
        for s in res["sources"]:
            assert s["language"] == "en"

def test_6_autodetect_telugu_query(rag_service: RAGService):
    res = rag_service.answer("కార్పొరేషన్ అంటే ఏమిటి?", language_code="Auto Detect")
    assert res["language_code"] in ("te-IN", "te")
    if res.get("sources"):
        for s in res["sources"]:
            assert s["language"] == "en"

def test_7_conversational_hello_returns_no_sources(rag_service: RAGService):
    res = rag_service.answer("hello", language_code="en-IN")
    assert res["grounded"] is False
    assert res["sources"] == []

def test_8_canonical_evidence_is_never_labeled_hindi(rag_service: RAGService):
    res = rag_service.answer("What is a corporation?", language_code="en-IN")
    for s in res.get("sources", []):
        assert s["language"] == "en"
        assert s["language"] != "hi"

def test_9_telugu_query_evidence_is_never_labeled_telugu(rag_service: RAGService):
    res = rag_service.answer("కార్పొరేషన్ అంటే ఏమిటి?", language_code="te-IN")
    for s in res.get("sources", []):
        assert s["language"] == "en"
        assert s["language"] != "te"

def test_10_unknown_question_refuses_hallucination(rag_service: RAGService):
    res = rag_service.answer("Who won the Mars Rovers Race in 2099?", language_code="en-IN")
    assert res["grounded"] is False
    assert res["sources"] == []
