from typing import Tuple

class InputGuardrail:
    def __init__(self, max_query_length: int = 1000):
        self.max_query_length = max_query_length

    def validate(self, query: str) -> Tuple[bool, str]:
        if not query or not query.strip():
            return False, "Query string cannot be empty or whitespace-only."

        clean_q = query.strip()
        if len(clean_q) > self.max_query_length:
            return False, f"Query exceeds maximum allowed length of {self.max_query_length} characters."

        # Check for obvious control char anomalies
        if any(ord(char) < 32 and char not in ("\n", "\r", "\t") for char in clean_q):
            return False, "Query contains invalid non-printable control characters."

        return True, ""
