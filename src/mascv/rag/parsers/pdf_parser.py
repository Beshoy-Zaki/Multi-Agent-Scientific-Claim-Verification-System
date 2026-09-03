"""PDF document parser extracting text, tables, and layout structure."""

from mascv.rag.parsers.base_parser import BaseDocumentParser
from mascv.models.paper import ResearchPaper


class PDFParser(BaseDocumentParser):
    """Parses academic PDF papers into structured content."""

    def parse(self, file_path: str) -> ResearchPaper:
        """Extract sections, text, and tables from PDF."""
        raise NotImplementedError("PDF parsing logic to be implemented.")
