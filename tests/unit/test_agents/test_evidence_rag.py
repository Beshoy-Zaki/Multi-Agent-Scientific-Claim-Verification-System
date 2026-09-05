"""Unit tests for EvidenceRAGAgent."""

from unittest.mock import patch

from mascv.agents.evidence_rag import EvidenceRAGAgent


def test_evidence_rag_initialization():
    with patch("mascv.agents.evidence_rag.EvidenceExtractor"):
        agent = EvidenceRAGAgent()

    assert agent.name == "EvidenceRAGAgent"