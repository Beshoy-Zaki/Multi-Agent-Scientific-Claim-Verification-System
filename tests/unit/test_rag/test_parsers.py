"""Unit tests for document parsers and section extractors."""

import os
import pytest
from mascv.rag.parsers.pdf_parser import PDFParser
from mascv.rag.parsers.section_extractor import SectionExtractor
from mascv.core.exceptions import DocumentParsingError


def test_pdf_parser_init():
    """Test PDFParser initialization."""
    parser = PDFParser()
    assert parser is not None
    assert parser.section_extractor is not None


def test_section_extractor_canonical_text():
    """Test that SectionExtractor partitions academic headers accurately."""
    extractor = SectionExtractor()
    sample_text = (
        "Title of Paper\n\n"
        "Abstract\nThis paper proposes a fast transformer.\n\n"
        "1 Introduction\nTransformers are widely used in NLP.\n\n"
        "2 Related Work\nPrior work includes RNNs and CNNs.\n\n"
        "3 Methodology\nOur method replaces softmax with linear attention.\n\n"
        "4 Conclusion\nIn conclusion, our method is 2x faster."
    )
    sections = extractor.extract_sections(sample_text)
    assert len(sections) >= 4
    titles = [s.title for s in sections]
    assert any("Abstract" in t for t in titles)
    assert any("Introduction" in t for t in titles)
    assert any("Methodology" in t for t in titles)
    assert any("Conclusion" in t for t in titles)


def test_pdf_parser_nonexistent_file():
    """PDFParser raises FileNotFoundError if target file does not exist."""
    parser = PDFParser()
    with pytest.raises(FileNotFoundError):
        parser.parse("non_existent_file.pdf")


def test_pdf_parser_on_sample_paper():
    """Test PDFParser on sample ingested PDF if available."""
    sample_path = os.path.join("data", "sample_inputs", "lora_2106.09685.pdf")
    if not os.path.exists(sample_path):
        pytest.skip("Sample PDF not present in data/sample_inputs")

    parser = PDFParser()
    paper = parser.parse(sample_path)

    assert paper.id == "lora_2106.09685"
    assert "LORA" in paper.metadata.title.upper()
    assert paper.metadata.arxiv_id is not None
    assert len(paper.raw_text) > 5000
    assert len(paper.sections) > 0
    assert paper.metadata.abstract is not None
