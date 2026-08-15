import re
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from app.core.config import settings, DATA_DIR

def normalize_text_for_matching(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[\.\?\!\,\;\:\'\"\\/\(\)\[\]\{\}\-\_\=\+]', ' ', text)
    tokens = text.strip().split()
    return " ".join(tokens)

class QueryAlignmentService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QueryAlignmentService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.exact_alignment_map: Dict[str, Dict[str, Any]] = {}
        self.query_id_to_eng: Dict[int, str] = {}
        self._initialized = True

    def initialize(self, chunks_path: Optional[Path] = None):
        if not chunks_path:
            chunks_path = getattr(settings, "CANONICAL_INDEX_DIR", DATA_DIR / "indexes" / "canonical") / "processed_chunks.json"

        if not chunks_path.exists():
            print(f"[QueryAlignmentService] Warning: processed_chunks.json missing at {chunks_path}")
            return

        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            count = 0
            for c in chunks:
                qid = c.get("query_id")
                eng_q = c.get("original_query") or ""
                trans_map = c.get("translated_queries") or {}

                if qid and eng_q:
                    self.query_id_to_eng[int(qid)] = eng_q

                norm_eng = normalize_text_for_matching(eng_q)
                if norm_eng and norm_eng not in self.exact_alignment_map:
                    self.exact_alignment_map[norm_eng] = {
                        "query_id": qid,
                        "canonical_english_query": eng_q,
                        "match_lang": "en"
                    }
                    count += 1

                for lang_code, trans_q in trans_map.items():
                    norm_trans = normalize_text_for_matching(trans_q)
                    if norm_trans and norm_trans not in self.exact_alignment_map:
                        self.exact_alignment_map[norm_trans] = {
                            "query_id": qid,
                            "canonical_english_query": eng_q,
                            "match_lang": lang_code
                        }
                        count += 1

            print(f"[QueryAlignmentService] Initialized dataset query alignment index with {count} query variants ({len(self.query_id_to_eng)} canonical queries).")
        except Exception as e:
            print(f"[QueryAlignmentService] Error building query alignment index: {e}")

    def align(self, query: str) -> Dict[str, Any]:
        if not self.exact_alignment_map:
            self.initialize()

        norm_q = normalize_text_for_matching(query)
        if not norm_q:
            return {
                "matched": False,
                "query_alignment": "none",
                "canonical_query_id": None,
                "aligned_english_query": None,
                "alignment_score": 0.0,
                "match_lang": None
            }

        # 1. Exact normalized match
        if norm_q in self.exact_alignment_map:
            res = self.exact_alignment_map[norm_q]
            return {
                "matched": True,
                "query_alignment": "dataset",
                "canonical_query_id": res["query_id"],
                "aligned_english_query": res["canonical_english_query"],
                "alignment_score": 1.0,
                "match_lang": res["match_lang"]
            }

        # 2. Substring or token overlap match fallback
        q_tokens = set(norm_q.split())
        best_match = None
        best_score = 0.0

        if len(q_tokens) >= 2:
            for k, val in self.exact_alignment_map.items():
                k_tokens = set(k.split())
                if not k_tokens:
                    continue
                intersection = q_tokens.intersection(k_tokens)
                union = q_tokens.union(k_tokens)
                jaccard = len(intersection) / float(len(union))
                if jaccard > best_score and jaccard >= 0.65:
                    best_score = jaccard
                    best_match = val

        if best_match and best_score >= 0.65:
            return {
                "matched": True,
                "query_alignment": "dataset_fuzzy",
                "canonical_query_id": best_match["query_id"],
                "aligned_english_query": best_match["canonical_english_query"],
                "alignment_score": round(best_score, 2),
                "match_lang": best_match["match_lang"]
            }

        return {
            "matched": False,
            "query_alignment": "none",
            "canonical_query_id": None,
            "aligned_english_query": None,
            "alignment_score": 0.0,
            "match_lang": None
        }

def get_query_alignment_service() -> QueryAlignmentService:
    return QueryAlignmentService()
