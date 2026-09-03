"""Unit tests for EvidenceRAGAgent."""

import pytest
from mascv.agents.evidence_rag import EvidenceRAGAgent


def test_evidence_rag_initialization():
    agent = EvidenceRAGAgent()
    assert agent.name == "EvidenceRAGAgent"
