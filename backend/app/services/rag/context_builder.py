from typing import List, Dict, Any
from app.core.config import settings

class ContextBuilder:
    def __init__(self, max_chunks: int = None):
        self.max_chunks = max_chunks if max_chunks is not None else settings.MAX_CONTEXT_CHUNKS

    def build_context(self, chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return "NO CONTEXT AVAILABLE."

        top_chunks = chunks[:self.max_chunks]
        context_parts = []
        for idx, chunk in enumerate(top_chunks, start=1):
            cid = chunk.get("chunk_id", f"c_{idx}")
            text = chunk.get("text", "").strip()
            score = chunk.get("score", 0.0)
            
            part = (
                f"[SOURCE {idx}] (ID: {cid}, Relevance Score: {score:.2f})\n"
                f"{text}"
            )
            context_parts.append(part)

        return "\n\n".join(context_parts)
