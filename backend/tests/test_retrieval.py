import pytest
import numpy as np
from app.services.retrieval.embeddings import EmbeddingService
from app.services.retrieval.vector_store import FAISSVectorStore
from app.services.retrieval.bm25_retriever import BM25Retriever, tokenize_text
from app.services.retrieval.hybrid_retriever import HybridRetriever, min_max_normalize
from app.services.retrieval.reranker import RerankerService
from app.services.retrieval.retrieval_service import RetrievalService

@pytest.fixture(scope="module")
def mock_chunks():
    return [
        {
            "chunk_id": "c1",
            "text": "विटामिन बी का अत्यधिक सेवन स्वास्थ्य के लिए हानिकारक हो सकता है।",
            "query_id": 1,
            "is_selected": 1,
            "metadata": {"source": "doc1"}
        },
        {
            "chunk_id": "c2",
            "text": "विटामिन बी12 शरीर में लाल रक्त कोशिकाओं के निर्माण में मदद करता है।",
            "query_id": 1,
            "is_selected": 0,
            "metadata": {"source": "doc2"}
        },
        {
            "chunk_id": "c3",
            "text": "Excessive intake of B vitamins may cause liver toxicity and nerve damage.",
            "query_id": 2,
            "is_selected": 1,
            "metadata": {"source": "doc3"}
        },
        {
            "chunk_id": "c4",
            "text": "भारत की राजधानी नई दिल्ली है और यहाँ ऐतिहासिक धरोहरें हैं।",
            "query_id": 3,
            "is_selected": 1,
            "metadata": {"source": "doc4"}
        }
    ]

def test_tokenization():
    text = "विटामिन B12 स्वास्थ्य के लिए 100% उपयोगी है!"
    tokens = tokenize_text(text)
    assert "विटामिन" in tokens
    assert "b12" in tokens
    assert "100" in tokens

def test_min_max_normalize():
    scores = [10.0, 20.0, 30.0]
    norm = min_max_normalize(scores)
    assert norm[0] == 0.0
    assert norm[1] == 0.5
    assert norm[2] == 1.0

    # Constant scores
    norm_same = min_max_normalize([5.0, 5.0])
    assert norm_same == [1.0, 1.0]

def test_faiss_vector_store(mock_chunks):
    store = FAISSVectorStore()
    dim = 384
    np.random.seed(42)
    fake_embeddings = np.random.randn(len(mock_chunks), dim).astype(np.float32)
    
    store.build_index(fake_embeddings, mock_chunks)
    assert store.is_loaded()
    
    query_vec = fake_embeddings[0]
    results = store.search(query_vec, top_k=2)
    assert len(results) == 2
    assert results[0]["chunk_id"] == "c1"
    assert "dense_score" in results[0]

def test_bm25_retriever(mock_chunks):
    bm25 = BM25Retriever()
    bm25.build_index(mock_chunks)
    
    results = bm25.search("विटामिन बी हानिकारक", top_k=2)
    assert len(results) >= 1
    assert results[0]["chunk_id"] in ["c1", "c2"]
    assert "bm25_score" in results[0]

def test_hybrid_retriever_fusion(mock_chunks):
    hybrid = HybridRetriever(dense_weight=0.7, bm25_weight=0.3)
    dense_res = [
        {"chunk_id": "c1", "text": mock_chunks[0]["text"], "dense_score": 0.9, "metadata": mock_chunks[0]["metadata"]},
        {"chunk_id": "c2", "text": mock_chunks[1]["text"], "dense_score": 0.5, "metadata": mock_chunks[1]["metadata"]}
    ]
    bm25_res = [
        {"chunk_id": "c1", "text": mock_chunks[0]["text"], "bm25_score": 5.0, "metadata": mock_chunks[0]["metadata"]},
        {"chunk_id": "c3", "text": mock_chunks[2]["text"], "bm25_score": 3.0, "metadata": mock_chunks[2]["metadata"]}
    ]
    
    fused = hybrid.fuse(dense_res, bm25_res, top_k=5)
    assert len(fused) == 3
    # Candidate c1 present in both should rank highest
    assert fused[0]["chunk_id"] == "c1"
    assert "hybrid_score" in fused[0]
    assert fused[0]["metadata"]["source"] == "doc1"

def test_retrieval_service_empty_query():
    service = RetrievalService()
    with pytest.raises(ValueError, match="Query string cannot be empty"):
        service.retrieve("")
        
    with pytest.raises(ValueError, match="Query string cannot be empty"):
        service.retrieve("   ")
