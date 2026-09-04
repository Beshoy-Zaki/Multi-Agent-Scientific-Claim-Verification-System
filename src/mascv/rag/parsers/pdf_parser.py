"""PDF document parser extracting text, tables, and layout structure."""

import os
import re
from typing import List, Optional
import pypdf

from mascv.core.exceptions import DocumentParsingError
from mascv.models.paper import DocumentSection, PaperMetadata, ResearchPaper
from mascv.rag.parsers.base_parser import BaseDocumentParser
from mascv.rag.parsers.section_extractor import SectionExtractor
from mascv.utils.logger import get_logger

logger = get_logger(__name__)


class PDFParser(BaseDocumentParser):
    """Parses academic PDF papers into structured ResearchPaper content."""

    def __init__(self, section_extractor: Optional[SectionExtractor] = None) -> None:
        self.section_extractor = section_extractor or SectionExtractor()

    def parse(self, file_path: str) -> ResearchPaper:
        """Extract sections, text, and metadata from an academic PDF."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at path: '{file_path}'")

        if not os.path.isfile(file_path):
            raise DocumentParsingError(f"Path is not a regular file: '{file_path}'")

        try:
            reader = pypdf.PdfReader(file_path)
        except Exception as exc:
            raise DocumentParsingError(f"Failed to open PDF file '{file_path}': {exc}") from exc

        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:
                raise DocumentParsingError(f"Encrypted PDF cannot be read: {exc}") from exc

        # 1. Extract text from all pages
        page_texts: List[str] = []
        for page_idx, page in enumerate(reader.pages):
            try:
                extracted = page.extract_text() or ""
                page_texts.append(extracted)
            except Exception as exc:
                logger.warning(f"Failed extracting text from page {page_idx + 1}: {exc}")

        raw_text = "\n\n".join(page_texts).strip()
        if not raw_text:
            raise DocumentParsingError(f"No extractable text found in PDF: '{file_path}'")

        # 2. Clean common academic PDF artifacts (e.g. line-break hyphenation: "experi-\nment" -> "experiment")
        cleaned_text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", raw_text)

        # 3. Extract Metadata
        paper_id = os.path.splitext(os.path.basename(file_path))[0]
        metadata = self._extract_metadata(reader, cleaned_text, file_path)

        # 4. Extract Sections
        sections = self.section_extractor.extract_sections(cleaned_text)

        logger.info(
            f"Successfully parsed PDF '{paper_id}': {len(reader.pages)} pages, "
            f"{len(sections)} sections, {len(cleaned_text)} characters."
        )

        return ResearchPaper(
            id=paper_id,
            metadata=metadata,
            raw_text=cleaned_text,
            sections=sections,
            tables_and_figures=[],
        )

    def _extract_metadata(
        self, reader: pypdf.PdfReader, text: str, file_path: str
    ) -> PaperMetadata:
        """Infer bibliographic metadata from PDF document info and header text."""
        pdf_info = reader.metadata or {}

        # Title detection
        doc_title = pdf_info.get("/Title")
        if doc_title and str(doc_title).strip() and not str(doc_title).lower().startswith("untitled"):
            title = str(doc_title).strip()
        else:
            # Infer title from first non-empty lines of page 1
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            title = lines[0] if lines else os.path.splitext(os.path.basename(file_path))[0]

        # Author detection
        authors: List[str] = []
        doc_author = pdf_info.get("/Author")
        if doc_author and str(doc_author).strip():
            authors = [a.strip() for a in str(doc_author).split(",") if a.strip()]

        # arXiv ID detection (e.g. "arXiv:2106.09685" or "2106.09685v2")
        arxiv_match = re.search(r"arxiv:\s*(\d{4}\.\d{4,5}(?:v\d+)?)", text, re.IGNORECASE)
        arxiv_id = arxiv_match.group(1) if arxiv_match else None

        # DOI detection (e.g. "10.1145/1234567.1234568")
        doi_match = re.search(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b", text)
        doi = doi_match.group(1) if doi_match else None

        # Abstract extraction
        abstract = self._extract_abstract(text)

        return PaperMetadata(
            title=title,
            authors=authors,
            abstract=abstract,
            arxiv_id=arxiv_id,
            doi=doi,
        )

    def _extract_abstract(self, text: str) -> Optional[str]:
        """Extract the abstract block from the beginning of the text."""
        match = re.search(
            r"(?:^|\n)\s*Abstract\s*[:\.\-]?\s*(.*?)(?=\n\s*(?:1\.?\s+|I\.?\s+)?Introduction|\n\s*1\s+[A-Z]|\Z)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        return None
