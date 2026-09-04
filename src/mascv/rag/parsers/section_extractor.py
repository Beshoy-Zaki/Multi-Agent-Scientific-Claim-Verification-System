"""Extracts and classifies academic paper sections."""

import re
from typing import List, Tuple
from mascv.models.paper import DocumentSection
from mascv.utils.logger import get_logger

logger = get_logger(__name__)


class SectionExtractor:
    """Identifies paper structure (Abstract, Methodology, Experiments, Limitations)."""

    SECTION_HEADER_REGEX = re.compile(
        r"(?:\n|^)\s*"
        r"(?:(?:\d+\.?(?:\d+)?|[I|V|X|L|C|D|M]+\.)\s+)?"
        r"(Abstract|Introduction|Background|Related Work|Literature Review|"
        r"Methodology|Method|Methods|Proposed Method|Model Architecture|"
        r"Experiments|Experimental Setup|Experimental Results|Results|Evaluation|"
        r"Discussion|Ablation Studies|Limitations|Conclusion|Conclusions|Future Work|References)"
        r"[^\n]*",
        re.IGNORECASE,
    )

    def extract_sections(self, raw_text: str) -> List[DocumentSection]:
        """Classify and partition document text by academic sections."""
        if not raw_text or not raw_text.strip():
            return []

        matches = list(self.SECTION_HEADER_REGEX.finditer(raw_text))
        if not matches:
            # If no standard headers found, return entire text as one section
            return [DocumentSection(title="Body", content=raw_text.strip())]

        sections: List[DocumentSection] = []

        # Check if there is introductory text before the first detected header
        if matches[0].start() > 0:
            preamble = raw_text[: matches[0].start()].strip()
            if preamble:
                sections.append(DocumentSection(title="Preamble", content=preamble))

        # Extract content between matched headers
        for i, match in enumerate(matches):
            title = match.group(0).strip()
            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)

            content = raw_text[start_pos:end_pos].strip()
            sections.append(DocumentSection(title=title, content=content))

        logger.debug(f"Extracted {len(sections)} sections from text.")
        return sections
