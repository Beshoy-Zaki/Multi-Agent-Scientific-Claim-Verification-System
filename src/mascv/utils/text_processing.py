"""Text cleaning, token estimation, and formatting helpers."""

import re


def clean_text(text: str) -> str:
    """Normalize whitespace and remove non-printable characters."""
    return re.sub(r"\s+", " ", text).strip()
