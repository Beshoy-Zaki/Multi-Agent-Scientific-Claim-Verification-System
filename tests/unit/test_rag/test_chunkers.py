"""Unit tests for chunkers."""

import pytest
from mascv.rag.chunking.agentic_chunker import AgenticChunker
from mascv.rag.chunking.fixed_chunker import FixedSizeChunker


def test_chunkers_init():
    agentic = AgenticChunker()
    fixed = FixedSizeChunker()
    assert agentic is not None
    assert fixed.chunk_size == 512
