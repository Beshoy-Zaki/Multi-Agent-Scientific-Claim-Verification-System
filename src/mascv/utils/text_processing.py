"""Text cleaning, token estimation, and formatting helpers."""

import re


def clean_text(text: str) -> str:
    """Normalize whitespace and remove non-printable characters."""
    return re.sub(r"\s+", " ", text).strip()


def extract_json_from_text(text: str) -> str:
    """Extract valid JSON substring from LLM output, handling markdown fences and thought scratchpads."""
    if not text:
        return ""

    # 1. If wrapped in markdown ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*([\[\{][\s\S]*?[\]\}])\s*```", text)
    if fence_match:
        return fence_match.group(1).strip()

    # 2. Find outermost JSON array [...]
    first_bracket = text.find("[")
    last_bracket = text.rfind("]")
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        return text[first_bracket : last_bracket + 1].strip()

    # 3. Find outermost JSON object {...}
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace : last_brace + 1].strip()

    return text.strip()
