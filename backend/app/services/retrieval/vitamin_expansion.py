import re
from typing import Dict, Any, List, Tuple

VITAMIN_SYNONYMS = {
    "b1": ["b1", "b-1", "thiamine", "thiamin"],
    "b2": ["b2", "b-2", "riboflavin"],
    "b3": ["b3", "b-3", "niacin", "niacinamide"],
    "b5": ["b5", "b-5", "pantothenic acid"],
    "b6": ["b6", "b-6", "pyridoxine"],
    "b7": ["b7", "b-7", "biotin"],
    "b9": ["b9", "b-9", "folate", "folic acid"],
    "b12": ["b12", "b-12", "cobalamin"],
}

TERMINOLOGY_ONLY_CHUNK_IDS = {"p_1090320_4"}

def extract_query_intent(query: str) -> str:
    if not query:
        return "general"
    q_lower = query.lower()
    benefit_keywords = [
        "benefit", "benefits", "good for", "helps", "help", "function", "functions",
        "role", "roles", "effect", "effects", "advantage", "advantages", "use", "uses",
        "action", "actions", "why take", "what does", "how does"
    ]
    for kw in benefit_keywords:
        if re.search(r'\b' + re.escape(kw) + r'\b', q_lower) or kw in q_lower:
            return "benefits"
    def_keywords = ["what is", "what are", "define", "meaning", "definition", "which vitamin", "identity"]
    for kw in def_keywords:
        if kw in q_lower:
            return "definition"
    return "general"

def detect_vitamin_terms(query: str) -> List[Tuple[str, str]]:
    """Returns list of (matched_text, vitamin_key) e.g. [('vitamin B2', 'b2')]"""
    if not query:
        return []
    q_lower = query.lower()
    matches = []
    
    # Pattern for Vitamin B1..B12 or B-1..B-12
    pattern = r'\b(vitamin\s+)?b-?(12|1|2|3|5|6|7|9)\b'
    for m in re.finditer(pattern, q_lower):
        v_num = m.group(2)
        v_key = f"b{v_num}"
        matches.append((m.group(0), v_key))
    
    # Standalone chemical names
    chem_map = {
        "thiamine": "b1", "thiamin": "b1",
        "riboflavin": "b2",
        "niacin": "b3", "niacinamide": "b3",
        "pantothenic": "b5",
        "pyridoxine": "b6",
        "biotin": "b7",
        "folate": "b9", "folic acid": "b9",
        "cobalamin": "b12"
    }
    for chem, v_key in chem_map.items():
        if re.search(r'\b' + re.escape(chem) + r'\b', q_lower):
            if not any(k == v_key for _, k in matches):
                matches.append((chem, v_key))
    return matches

def expand_vitamin_query(query: str) -> Dict[str, Any]:
    matched_vitamins = detect_vitamin_terms(query)
    intent = extract_query_intent(query)
    
    if not matched_vitamins:
        return {
            "has_vitamin": False,
            "normalized_query": query.strip() if query else "",
            "expanded_queries": [query.strip()] if query else [],
            "intent": intent,
            "vitamin_keys": []
        }
        
    v_keys = list(set([k for _, k in matched_vitamins]))
    expanded = [query.strip()]
    
    for v_key in v_keys:
        syns = VITAMIN_SYNONYMS.get(v_key, [])
        for syn in syns:
            expanded.append(syn)
            expanded.append(f"vitamin {syn}")
            if syn.startswith("b") and len(syn) > 1 and syn[1:].isdigit():
                expanded.append(f"vitamin {syn[0]}-{syn[1:]}")
                expanded.append(f"b-{syn[1:]}")
            if intent == "benefits":
                expanded.append(f"{syn} benefits")
                expanded.append(f"benefits of {syn}")
                    
    seen = set()
    dedup_expanded = []
    for q in expanded:
        q_clean = q.strip()
        if q_clean and q_clean.lower() not in seen:
            seen.add(q_clean.lower())
            dedup_expanded.append(q_clean)
            
    return {
        "has_vitamin": True,
        "normalized_query": query.strip(),
        "expanded_queries": dedup_expanded,
        "intent": intent,
        "vitamin_keys": v_keys
    }

def does_chunk_support_intent(chunk: Dict[str, Any], query_intent: str, query: str = "") -> Tuple[bool, str]:
    """
    Checks if candidate chunk actually supports the user's intent.
    - Ingredient lists or terminology definitions (like p_1090320_4) support "definition", but NOT "benefits".
    """
    chunk_id = chunk.get("chunk_id", "")
    text = chunk.get("text", "").lower()
    
    if query_intent != "benefits":
        return True, "Chunk supports general/definition intent"
        
    # If query intent is benefits, check if chunk is known terminology-only chunk
    if chunk_id in TERMINOLOGY_ONLY_CHUNK_IDS:
        return False, f"Chunk {chunk_id} is an ingredient list/terminology source with no health benefit content"
        
    # If text is pure ingredient / supplement formula listing like "Vitamin B-2 100 mg", reject for benefits
    is_ingredient_listing = bool(re.search(r'\b(mg|mcg|g|iu)\b', text)) and not any(
        kw in text for kw in ["prevents", "treats", "treatment of", "required for", "essential for", "deficiency", "promotes", "maintains", "body needs"]
    )
    if is_ingredient_listing and ("formula" in text or "ingredient" in text or "supplement" in text or text.count("mg") >= 2):
        return False, f"Chunk {chunk_id} contains product/ingredient specs without health benefit details"

    # For vitamin benefit query, verify text mentions target vitamin/compound
    if query:
        matched_vits = detect_vitamin_terms(query)
        if matched_vits:
            vit_synonyms = []
            for _, v_key in matched_vits:
                vit_synonyms.extend(VITAMIN_SYNONYMS.get(v_key, []))
            has_target_mention = any(re.search(r'\b' + re.escape(s) + r'\b', text) for s in vit_synonyms)
            if not has_target_mention:
                return False, f"Chunk {chunk_id} does not mention target vitamin for benefit query"

    # Specific health benefit evidence signals
    specific_benefit_signals = [
        "prevents", "treats", "treatment", "cures", "essential for", "required for",
        "helps with", "helps to", "role in", "functions to", "deficiency causes",
        "deficiency leads to", "supports health", "promotes", "maintains", "vital for",
        "health benefits of", "reduces risk", "improves"
    ]
    if any(sig in text for sig in specific_benefit_signals):
        return True, "Chunk contains explicit health benefit evidence"

    return False, f"Chunk {chunk_id} lacks explicit health benefit evidence for the requested query"
