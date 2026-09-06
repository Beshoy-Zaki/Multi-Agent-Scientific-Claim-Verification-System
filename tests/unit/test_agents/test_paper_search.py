"""Unit tests for PaperSearchAgent."""

import pytest
from unittest.mock import MagicMock
from mascv.agents.paper_search import PaperSearchAgent


def test_paper_search_initialization():
    agent = PaperSearchAgent()
    assert agent.name == "PaperSearchAgent"
    assert len(agent.search_clients) >= 1
    assert agent.max_papers_per_claim == 5


def test_paper_search_heuristic_queries():
    agent = PaperSearchAgent()
    queries = agent.generate_adversarial_queries("LoRA reduces trainable parameters")
    assert "supporting_queries" in queries
    assert "adversarial_queries" in queries
    assert len(queries["supporting_queries"]) >= 1
    assert len(queries["adversarial_queries"]) >= 1


def test_paper_search_with_mock_llm():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = """
    {
      "claim_id": "C1",
      "search_queries": {
        "supporting_queries": ["LoRA benchmark replication", "LoRA memory efficiency"],
        "adversarial_queries": ["LoRA performance degradation", "LoRA catastrophic forgetting"]
      }
    }
    """

    agent = PaperSearchAgent(llm_client=mock_llm)
    queries = agent.generate_adversarial_queries("LoRA reduces trainable parameters")

    assert len(queries["supporting_queries"]) == 2
    assert "LoRA benchmark replication" in queries["supporting_queries"]
    assert "LoRA performance degradation" in queries["adversarial_queries"]
