"""PDF parsing utilities for scientific papers."""

from pathlib import Path
from typing import List, Dict, Any

from pypdf import PdfReader


def parse_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from a PDF page by page.

    Each returned item contains:
    - page number
    - page text
    """

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(path))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append(
                {
                    "page_number": page_number,
                    "text": text.strip(),
                }
            )

    return pages