"""Unit tests for hybrid retriever and reranker."""

import pytest
from mascv.rag.retriever.hybrid_retriever import HybridRetriever
from mascv.rag.retriever.reranker import EvidenceReranker


def test_retriever_init():
    retriever = HybridRetriever()
    reranker = EvidenceReranker()
    assert retriever is not None
    assert reranker is not None
