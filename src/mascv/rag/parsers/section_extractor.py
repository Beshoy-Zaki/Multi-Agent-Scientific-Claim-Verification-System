"""Extracts and classifies academic paper sections."""

from typing import List
from mascv.models.paper import DocumentSection


class SectionExtractor:
    """Identifies paper structure (Abstract, Methodology, Experiments, Limitations)."""

    def extract_sections(self, raw_text: str) -> List[DocumentSection]:
        """Classify and partition document text by academic sections."""
        raise NotImplementedError("Section extraction logic to be implemented.")
