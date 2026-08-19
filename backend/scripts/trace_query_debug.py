import sys
import os
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.services.rag.rag_service import get_rag_service

def trace(query: str, lang: str):
    print("="*80)
    print(f"TRACING QUERY: '{query}' (Requested Lang: '{lang}')")
    print("="*80)
    
    rag = get_rag_service()
    rag.initialize(load_indexes=True)

    # 1. Retrieval inspect
    ret_res = rag.retrieval_service.retrieve(query=query, language_code=lang)
    chunks = ret_res.get("results", [])
    print(f"\n[RETRIEVAL COUNT]: {len(chunks)}")
    for idx, c in enumerate(chunks[:5], 1):
        print(f" Chunk {idx}: ID={c['chunk_id']}, Score={c['score']}, Dense={c['dense_score']}, BM25={c['bm25_score']}")
        print(f"  Text: {c['text'][:120]}...")

    # 2. Retrieval Guardrail inspect
    suff, score, msg = rag.retrieval_guardrail.evaluate(chunks, query=query, language_code=lang)
    print(f"\n[RETRIEVAL GUARDRAIL]: Sufficient={suff}, TopScore={score}, RefusalMsg='{msg}'")

    # 3. LLM Call inspect
    clean_chunks = rag.prompt_injection_guardrail.sanitize_chunks(chunks)
    context_str = rag.context_builder.build_context(clean_chunks)
    prompt_str = rag.prompt_builder.build_prompt(query, context_str, language_code=lang)
    
    llm_res = rag.answer_generator.generate(query, prompt_str, clean_chunks, language_code=lang)
    print(f"\n[LLM GENERATOR OUTPUT]:")
    print(f" Answer: {llm_res.get('answer')}")
    print(f" Grounded: {llm_res.get('grounded')}")
    print(f" Confidence: {llm_res.get('confidence')}")
    print(f" LLM Mode: {llm_res.get('llm_mode')}")

    # 4. Grounding Validator inspect
    is_gr, gr_conf, fin_ans = rag.grounding_validator.validate(llm_res.get("answer", ""), clean_chunks, query, language_code=lang)
    print(f"\n[GROUNDING VALIDATOR]:")
    print(f" Is Grounded: {is_gr}")
    print(f" Confidence: {gr_conf}")
    print(f" Final Answer: {fin_ans}")

    # 5. Full RAG Service Answer
    full_res = rag.answer(query=query, language_code=lang)
    print(f"\n[FULL RAG ANSWER RESULT]:")
    print(f" Answer: {full_res['answer']}")
    print(f" Grounded: {full_res['grounded']}")
    print(f" Confidence: {full_res['confidence']}")
    print(f" Sources count: {len(full_res['sources'])}")
    print("="*80)

if __name__ == "__main__":
    trace("What is a corporation?", "en-IN")
    trace("कॉर्पोरेशन क्या है?", "hi-IN")
    trace("కార్పొరేషన్ అంటే ఏమిటి?", "te-IN")
