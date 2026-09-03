"""Unit tests for document parsers."""

import pytest
from mascv.rag.parsers.pdf_parser import PDFParser


def test_pdf_parser_init():
    parser = PDFParser()
    assert parser is not None
