from typing import List, Dict, Any
from app.core.config import settings

def min_max_normalize(scores: List[float]) -> List[float]:
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    range_s = max_s - min_s
    if range_s <= 1e-6:
        return [1.0 for _ in scores]
    return [(s - min_s) / range_s for s in scores]

class HybridRetriever:
    def __init__(self, dense_weight: float = None, bm25_weight: float = None):
        self.dense_weight = dense_weight if dense_weight is not None else settings.DENSE_WEIGHT
        self.bm25_weight = bm25_weight if bm25_weight is not None else settings.BM25_WEIGHT

    def fuse(
        self,
        dense_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        # Normalize dense scores
        dense_raw_scores = [r["dense_score"] for r in dense_results]
        dense_norm_scores = min_max_normalize(dense_raw_scores)
        
        # Normalize BM25 scores
        bm25_raw_scores = [r["bm25_score"] for r in bm25_results]
        bm25_norm_scores = min_max_normalize(bm25_raw_scores)
        
        candidates: Dict[str, Dict[str, Any]] = {}
        
        # Add dense candidates
        for item, norm_s in zip(dense_results, dense_norm_scores):
            cid = item["chunk_id"]
            candidates[cid] = {
                "chunk_id": cid,
                "text": item["text"],
                "dense_score": item["dense_score"],
                "norm_dense_score": norm_s,
                "bm25_score": 0.0,
                "norm_bm25_score": 0.0,
                "metadata": item.get("metadata", {})
            }
            
        # Add / update BM25 candidates
        for item, norm_s in zip(bm25_results, bm25_norm_scores):
            cid = item["chunk_id"]
            if cid in candidates:
                candidates[cid]["bm25_score"] = item["bm25_score"]
                candidates[cid]["norm_bm25_score"] = norm_s
            else:
                candidates[cid] = {
                    "chunk_id": cid,
                    "text": item["text"],
                    "dense_score": 0.0,
                    "norm_dense_score": 0.0,
                    "bm25_score": item["bm25_score"],
                    "norm_bm25_score": norm_s,
                    "metadata": item.get("metadata", {})
                }
                
        # Calculate hybrid fusion score
        fused_list = []
        for cid, cand in candidates.items():
            hybrid_score = (
                self.dense_weight * cand["norm_dense_score"] +
                self.bm25_weight * cand["norm_bm25_score"]
            )
            cand["hybrid_score"] = float(hybrid_score)
            cand["score"] = float(hybrid_score)
            fused_list.append(cand)
            
        # Sort by hybrid score descending
        fused_list.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return fused_list[:top_k]
