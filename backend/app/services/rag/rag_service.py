import time
from typing import Dict, Any, List, Optional
from app.services.retrieval.retrieval_service import get_retrieval_service, RetrievalService
from app.services.guardrails.input_guardrail import InputGuardrail
from app.services.guardrails.retrieval_guardrail import RetrievalGuardrail
from app.services.guardrails.prompt_injection_guardrail import PromptInjectionGuardrail
from app.services.guardrails.output_guardrail import OutputGuardrail
from app.services.rag.context_builder import ContextBuilder
from app.services.rag.prompt_builder import PromptBuilder
from app.services.rag.answer_generator import get_answer_generator, AnswerGenerator
from app.services.rag.grounding_validator import GroundingValidator

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

        self.input_guardrail = InputGuardrail()
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
        print("[RAGService] Initializing RAG pipeline components...")
        self.retrieval_service = get_retrieval_service()
        self.retrieval_service.initialize(load_indexes=load_indexes)
        self.answer_generator = get_answer_generator()
        print("[RAGService] Initialization complete.")

    def answer(self, query: str) -> Dict[str, Any]:
        start_total = time.perf_counter()
        
        # 1. Input Guardrail
        t0 = time.perf_counter()
        valid, input_err = self.input_guardrail.validate(query)
        if not valid:
            total_lat = (time.perf_counter() - start_total) * 1000.0
            return {
                "query": query,
                "answer": input_err,
                "grounded": False,
                "confidence": 0.0,
                "sources": [],
                "latency": {
                    "retrieval_ms": 0.0,
                    "context_ms": 0.0,
                    "llm_ms": 0.0,
                    "grounding_ms": 0.0,
                    "total_ms": round(total_lat, 2)
                }
            }
        t_input = (time.perf_counter() - t0) * 1000.0

        # 2. Retrieval Service
        if self.retrieval_service is None:
            self.retrieval_service = get_retrieval_service()
            
        t0 = time.perf_counter()
        retrieval_res = self.retrieval_service.retrieve(query=query.strip())
        t_retrieval = (time.perf_counter() - t0) * 1000.0

        raw_chunks = retrieval_res.get("results", [])

        # 3. Retrieval Confidence Check
        sufficient, top_score, ref_msg = self.retrieval_guardrail.evaluate(raw_chunks)
        if not sufficient:
            total_lat = (time.perf_counter() - start_total) * 1000.0
            return {
                "query": query,
                "answer": ref_msg,
                "grounded": False,
                "confidence": round(top_score, 2),
                "sources": raw_chunks[:3],
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

        # 6. Prompt Construction
        prompt_str = self.prompt_builder.build_prompt(query, context_str)
        t_context = (time.perf_counter() - t0) * 1000.0

        # 7. LLM Call
        t0 = time.perf_counter()
        if self.answer_generator is None:
            self.answer_generator = get_answer_generator()
        llm_res = self.answer_generator.generate(query, prompt_str, clean_chunks)
        t_llm = (time.perf_counter() - t0) * 1000.0

        raw_answer = llm_res.get("answer", "")
        raw_grounded = llm_res.get("grounded", True)
        raw_conf = llm_res.get("confidence", 0.85)

        # 8. Output Guardrail
        valid_out, out_err = self.output_guardrail.validate(raw_answer)
        if not valid_out:
            total_lat = (time.perf_counter() - start_total) * 1000.0
            return {
                "query": query,
                "answer": f"Refused: {out_err}",
                "grounded": False,
                "confidence": 0.0,
                "sources": clean_chunks[:3],
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
        is_grounded, grounding_conf, final_answer = self.grounding_validator.validate(raw_answer, clean_chunks)
        t_grounding = (time.perf_counter() - t0) * 1000.0

        total_lat = (time.perf_counter() - start_total) * 1000.0

        # Format sources
        formatted_sources = []
        for c in clean_chunks[:5]:
            formatted_sources.append({
                "chunk_id": c["chunk_id"],
                "score": round(float(c.get("score", 0.0)), 4),
                "text": c["text"]
            })

        return {
            "query": query,
            "answer": final_answer,
            "grounded": is_grounded,
            "confidence": round(float(grounding_conf if is_grounded else raw_conf), 2),
            "sources": formatted_sources,
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
