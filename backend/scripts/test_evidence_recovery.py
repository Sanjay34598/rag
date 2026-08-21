import os
import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.rag.answer_generator import AnswerGenerator
from app.services.rag.grounding_validator import GroundingValidator
from app.services.retrieval.vitamin_expansion import does_chunk_support_intent, extract_query_intent

def run_tests():
    ag = AnswerGenerator()
    gv = GroundingValidator()
    
    print("==================================================")
    print("TESTING EVIDENCE RECOVERY & BENEFIT INTENT & CROSS-LINGUAL GROUNDING")
    print("==================================================")
    
    test_cases = [
        {
            "name": "Case 1: 'What is vitamin B2?' with B2/Riboflavin context",
            "query": "What is vitamin B2?",
            "context": [{"chunk_id": "c1", "text": "Vitamin B-2 (Riboflavin) 100 mg.", "score": 0.90}],
            "lang": "en-IN",
            "expected_grounded": True,
            "expected_answer_contains": "Riboflavin"
        },
        {
            "name": "Case 2: 'What is riboflavin?' with B2/Riboflavin context",
            "query": "What is riboflavin?",
            "context": [{"chunk_id": "c1", "text": "Vitamin B-2 (Riboflavin) 100 mg.", "score": 0.90}],
            "lang": "en-IN",
            "expected_grounded": True,
            "expected_answer_contains": "Vitamin B-2"
        },
        {
            "name": "Case 3: 'What are the benefits of vitamin B2?' with ingredient-only context",
            "query": "What are the benefits of vitamin B2?",
            "context": [{"chunk_id": "c1", "text": "Vitamin B-2 (Riboflavin) 100 mg.", "score": 0.90}],
            "lang": "en-IN",
            "expected_grounded": False,
            "expected_answer_contains": "couldn't verify"
        },
        {
            "name": "Case 4: 'What does vitamin B2 do?' with ingredient-only context",
            "query": "What does vitamin B2 do?",
            "context": [{"chunk_id": "c1", "text": "Vitamin B-2 (Riboflavin) 100 mg.", "score": 0.90}],
            "lang": "en-IN",
            "expected_grounded": False,
            "expected_answer_contains": "couldn't verify"
        },
        {
            "name": "Case 5: Hindi B2 identity query with English evidence",
            "query": "विटामिन B-2 क्या है?",
            "context": [{"chunk_id": "c1", "text": "Vitamin B-2 (Riboflavin) 100 mg.", "score": 0.90}],
            "lang": "hi-IN",
            "expected_grounded": True,
            "expected_answer_contains": "राइबोफ्लेविन"
        },
        {
            "name": "Case 6: Telugu B2 identity query with English evidence",
            "query": "రిబోఫ్లావిన్ అంటే ఏమిటి?",
            "context": [{"chunk_id": "c1", "text": "Vitamin B-2 (Riboflavin) 100 mg.", "score": 0.90}],
            "lang": "te-IN",
            "expected_grounded": True,
            "expected_answer_contains": "రిబోఫ్లావిన్"
        }
    ]

    all_passed = True

    for idx, tc in enumerate(test_cases, 1):
        print(f"\n--- {tc['name']} ---")
        q = tc["query"]
        ctx = tc["context"]
        lang = tc["lang"]
        
        q_intent = extract_query_intent(q)
        supports_intent, intent_reason = does_chunk_support_intent(ctx[0], q_intent, query=q)
        print(f"  Intent: '{q_intent}' | supports_intent={supports_intent} | reason='{intent_reason}'")
        
        if not supports_intent:
            res = {
                "answer": "I couldn't verify that answer from the available context.",
                "grounded": False,
                "confidence": 0.0
            }
        else:
            res = ag.generate(q, "", ctx, language_code=lang)
            is_gr, conf, ans = gv.validate(res["answer"], ctx, query=q, language_code=lang)
            res["grounded"] = is_gr
            res["confidence"] = conf
            res["answer"] = ans

        grounded_pass = res["grounded"] == tc["expected_grounded"]
        ans_pass = tc["expected_answer_contains"].lower() in res["answer"].lower()
        
        status = "PASS" if (grounded_pass and ans_pass) else "FAIL"
        if not (grounded_pass and ans_pass):
            all_passed = False

        print(f"  Result  : Grounded={res['grounded']} | Mode={res.get('llm_mode', 'N/A')}")
        print(f"  Answer  : '{res['answer']}'")
        print(f"  Status  : [{status}]")

    print("\n--- Case 7: EXPLICIT TEST FOR REMOVAL OF CROSS-LINGUAL BYPASS ---")
    unsupported_hindi_claim = "विटामिन B2 ऊर्जा उत्पादन, त्वचा और आंखों के स्वास्थ्य के लिए आवश्यक है।"
    ingredient_ctx = [{"chunk_id": "c1", "text": "Vitamin B-2 (Riboflavin) 100 mg."}]
    is_gr, conf, ans = gv.validate(unsupported_hindi_claim, ingredient_ctx, query="विटामिन B-2 क्या है?", language_code="hi-IN")
    print(f"  Unsupported Hindi Claim: '{unsupported_hindi_claim}'")
    print(f"  Context                : '{ingredient_ctx[0]['text']}'")
    print(f"  Grounding Validation   : Grounded={is_gr} | Confidence={conf}")
    
    if not is_gr:
        print("  Status                 : [PASS] (Unsupported Hindi claim correctly REJECTED; old bypass is GONE!)")
    else:
        print("  Status                 : [FAIL] (Bypass still active!)")
        all_passed = False

    print("\n==================================================")
    if all_passed:
        print("ALL 7 TEST CASES PASSED SUCCESSFULLY!")
    else:
        print("SOME TEST CASES FAILED!")
    print("==================================================")
    return all_passed

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
