import re
from typing import List, Dict, Any

class PromptInjectionGuardrail:
    INJECTION_PATTERNS = [
        r"ignore\s+(previous|all)\s+instructions",
        r"disregard\s+(previous|above)\s+instructions",
        r"system\s*prompt\s*:",
        r"you\s+are\s+now\s+a",
        r"new\s+system\s+instruction"
    ]

    def sanitize_text(self, text: str) -> str:
        if not text:
            return ""
        
        sanitized = text
        for pattern in self.INJECTION_PATTERNS:
            sanitized = re.sub(pattern, "[UNTRUSTED_CONTENT_FILTERED]", sanitized, flags=re.IGNORECASE)
        return sanitized

    def sanitize_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        clean_chunks = []
        for c in chunks:
            item = dict(c)
            item["text"] = self.sanitize_text(c.get("text", ""))
            clean_chunks.append(item)
        return clean_chunks
