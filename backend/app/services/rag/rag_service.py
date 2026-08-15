import re
import time
from typing import Dict, Any, List, Optional
from app.services.retrieval.retrieval_service import get_retrieval_service, RetrievalService
from app.services.guardrails.input_guardrail import InputGuardrail
from app.services.guardrails.retrieval_guardrail import RetrievalGuardrail
from app.services.guardrails.prompt_injection_guardrail import PromptInjectionGuardrail
from app.services.guardrails.output_guardrail import OutputGuardrail
from app.services.guardrails.query_intent_guard import QueryIntentGuard
from app.services.rag.context_builder import ContextBuilder
from app.services.rag.prompt_builder import PromptBuilder
from app.services.rag.answer_generator import get_answer_generator, AnswerGenerator
from app.services.rag.grounding_validator import GroundingValidator

def detect_script_language(query: str) -> str:
    if not query:
        return "hi-IN"
    # Telugu script range
    if re.search(r'[\u0C00-\u0C7F]', query):
        return "te-IN"
    # Devanagari (Hindi) script range
    if re.search(r'[\u0900-\u097F]', query):
        return "hi-IN"
    # Default Latin script to English
    return "en-IN"

class RAGService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.is_ready: bool = False
        self.input_guardrail = InputGuardrail()
        self.query_intent_guard = QueryIntentGuard()
        self.retrieval_guardrail = RetrievalGuardrail()
        self.prompt_injection_guardrail = PromptInjectionGuardrail()
        self.output_guardrail = OutputGuardrail()
        self.context_builder = ContextBuilder()
        self.prompt_builder = PromptBuilder()
        self.grounding_validator = GroundingValidator()
        self.retrieval_service: Optional[RetrievalService] = None
        self.answer_generator: Optional[AnswerGenerator] = None
        self._initialized = True

    def initialize(self, load_indexes: bool = True) -> None:
        self.retrieval_service = get_retrieval_service()
        self.retrieval_service.initialize(load_indexes=load_indexes)
        self.answer_generator = get_answer_generator()
        self.is_ready = True
        print("[RAG READY] Voice RAG is ready")

    def answer(self, query: str, language_code: str = None) -> Dict[str, Any]:
        start_total = time.perf_counter()
        
        # LANGUAGE PRIORITY RULES (Requirement 3):
        # If explicit language selected: USE IT (do NOT run script detection).
        # If Auto Detect / unknown / missing: RUN script detection on query text.
        if language_code and language_code.strip().lower() not in ("unknown", "auto", "auto detect", "none"):
            effective_lang = language_code.strip()
        else:
            effective_lang = detect_script_language(query)

        try:
            print(f"[RAGService] query='{query}', selected_lang='{language_code}', effective_lang='{effective_lang}'")
        except Exception:
            pass
        
        # 1. Input Guardrail
        t0 = time.perf_counter()
        valid, input_err = self.input_guardrail.validate(query)
        if not valid:
            total_lat = (time.perf_counter() - start_total) * 1000.0
            print(f"[PERF] language={effective_lang} intent=input_error embedding=0.00ms faiss=0.00ms bm25=0.00ms fusion=0.00ms validation=0.00ms groq=0.00ms grounding=0.00ms total={total_lat:.2f}ms")
            return {
                "query": query,
                "answer": input_err,
                "grounded": False,
                "confidence": 0.0,
                "sources": [],
                "language_code": effective_lang,
                "latency": {
                    "retrieval_ms": 0.0,
                    "context_ms": 0.0,
                    "llm_ms": 0.0,
                    "grounding_ms": 0.0,
                    "total_ms": round(total_lat, 2)
                }
            }
        t_input = (time.perf_counter() - t0) * 1000.0

        # 1.5 Conversational Intent Check (BEFORE Retrieval)
        is_conv, conv_ans = self.query_intent_guard.evaluate(query, language_code=effective_lang)
        if is_conv:
            total_lat = (time.perf_counter() - start_total) * 1000.0
            print(f"[PERF] language={effective_lang} intent=conversational embedding=0.00ms faiss=0.00ms bm25=0.00ms fusion=0.00ms validation=0.00ms groq=0.00ms grounding=0.00ms total={total_lat:.2f}ms")
            return {
                "query": query,
                "answer": conv_ans,
                "grounded": False,
                "confidence": 0.0,
                "sources": [],
                "language_code": effective_lang,
                "latency": {
                    "retrieval_ms": 0.0,
                    "context_ms": 0.0,
                    "llm_ms": 0.0,
                    "grounding_ms": 0.0,
                    "total_ms": round(total_lat, 2)
                }
            }

        # 2. Retrieval Service (Language-Aware Retrieval)
        if self.retrieval_service is None:
            self.retrieval_service = get_retrieval_service()
            
        t0 = time.perf_counter()
        retrieval_res = self.retrieval_service.retrieve(query=query.strip(), language_code=effective_lang)
        t_retrieval = (time.perf_counter() - t0) * 1000.0
        lat_bd = retrieval_res.get("latency_breakdown", {})

        raw_chunks = retrieval_res.get("results", [])

        # 3. Retrieval Confidence Check (Language-Aware Localized Refusal Fast Path)
        sufficient, top_score, ref_msg = self.retrieval_guardrail.evaluate(raw_chunks, query=query, language_code=effective_lang)
        if not sufficient:
            total_lat = (time.perf_counter() - start_total) * 1000.0
            print(f"[PERF] language={effective_lang} intent=insufficient_evidence_refusal embedding={lat_bd.get('embedding_ms',0.0):.2f}ms faiss={lat_bd.get('faiss_ms',0.0):.2f}ms bm25={lat_bd.get('bm25_ms',0.0):.2f}ms fusion={lat_bd.get('fusion_ms',0.0):.2f}ms validation=0.00ms groq=0.00ms grounding=0.00ms total={total_lat:.2f}ms")
            return {
                "query": query,
                "answer": ref_msg,
                "grounded": False,
                "confidence": round(top_score, 2),
                "sources": [],
                "language_code": effective_lang,
                "latency": {
                    "retrieval_ms": round(t_retrieval, 2),
                    "context_ms": 0.0,
                    "llm_ms": 0.0,
                    "grounding_ms": 0.0,
                    "total_ms": round(total_lat, 2)
                }
            }

        # 4. Prompt Injection Protection & Sanitization
        t0 = time.perf_counter()
        clean_chunks = self.prompt_injection_guardrail.sanitize_chunks(raw_chunks)

        # 5. Context Building
        context_str = self.context_builder.build_context(clean_chunks)

        # 6. Prompt Construction (with multilingual target instruction)
        prompt_str = self.prompt_builder.build_prompt(query, context_str, language_code=effective_lang)
        t_context = (time.perf_counter() - t0) * 1000.0

        # 7. LLM Call
        t0 = time.perf_counter()
        if self.answer_generator is None:
            self.answer_generator = get_answer_generator()
        llm_res = self.answer_generator.generate(query, prompt_str, clean_chunks, language_code=effective_lang)
        t_llm = (time.perf_counter() - t0) * 1000.0

        raw_answer = llm_res.get("answer", "")
        raw_grounded = llm_res.get("grounded", True)
        raw_conf = llm_res.get("confidence", 0.85)

        # 8. Output Guardrail
        valid_out, out_err = self.output_guardrail.validate(raw_answer)
        if not valid_out:
            total_lat = (time.perf_counter() - start_total) * 1000.0
            print(f"[PERF] language={effective_lang} intent=output_error embedding={lat_bd.get('embedding_ms',0.0):.2f}ms faiss={lat_bd.get('faiss_ms',0.0):.2f}ms bm25={lat_bd.get('bm25_ms',0.0):.2f}ms fusion={lat_bd.get('fusion_ms',0.0):.2f}ms validation=0.00ms groq={t_llm:.2f}ms grounding=0.00ms total={total_lat:.2f}ms")
            return {
                "query": query,
                "answer": f"Refused: {out_err}",
                "grounded": False,
                "confidence": 0.0,
                "sources": [],
                "language_code": effective_lang,
                "latency": {
                    "retrieval_ms": round(t_retrieval, 2),
                    "context_ms": round(t_context, 2),
                    "llm_ms": round(t_llm, 2),
                    "grounding_ms": 0.0,
                    "total_ms": round(total_lat, 2)
                }
            }

        # 9. Grounding Validation
        t0 = time.perf_counter()
        is_grounded, grounding_conf, final_answer = self.grounding_validator.validate(raw_answer, clean_chunks, query, language_code=effective_lang)
        t_grounding = (time.perf_counter() - t0) * 1000.0

        total_lat = (time.perf_counter() - start_total) * 1000.0

        # Format sources with full metadata provenance
        formatted_sources = []
        if is_grounded:
            for c in clean_chunks[:5]:
                formatted_sources.append({
                    "chunk_id": c["chunk_id"],
                    "language": c.get("language", "en"),
                    "query_id": c.get("query_id", 0),
                    "score": round(float(c.get("score", 0.0)), 4),
                    "text": c["text"],
                    "source_lang": c.get("source_lang", "eng_Latn"),
                    "target_lang": c.get("target_lang", None)
                })

        print(f"[PERF] language={effective_lang} intent=grounded_rag embedding={lat_bd.get('embedding_ms',0.0):.2f}ms faiss={lat_bd.get('faiss_ms',0.0):.2f}ms bm25={lat_bd.get('bm25_ms',0.0):.2f}ms fusion={lat_bd.get('fusion_ms',0.0):.2f}ms validation=0.00ms groq={t_llm:.2f}ms grounding={t_grounding:.2f}ms total={total_lat:.2f}ms")

        return {
            "query": query,
            "answer": final_answer,
            "grounded": is_grounded,
            "confidence": round(float(grounding_conf if is_grounded else 0.0), 2),
            "sources": formatted_sources,
            "language_code": effective_lang,
            "latency": {
                "retrieval_ms": round(t_retrieval, 2),
                "context_ms": round(t_context, 2),
                "llm_ms": round(t_llm, 2),
                "grounding_ms": round(t_grounding, 2),
                "total_ms": round(total_lat, 2)
            }
        }

def get_rag_service() -> RAGService:
    return RAGService()
