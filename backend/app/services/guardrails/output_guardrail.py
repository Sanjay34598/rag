from typing import Tuple

class OutputGuardrail:
    def validate(self, answer: str) -> Tuple[bool, str]:
        if not answer:
            return False, "Generated response was empty."
        
        # Check for system prompt leaks
        if "<untrusted_context>" in answer or "SYSTEM INSTRUCTIONS" in answer:
            return False, "Generated answer contained internal structural markers."

        return True, ""
