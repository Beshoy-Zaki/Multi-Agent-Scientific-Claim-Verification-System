"""Unit tests for PaperSearchAgent."""

import pytest
from mascv.agents.paper_search import PaperSearchAgent


def test_paper_search_initialization():
    agent = PaperSearchAgent()
    assert agent.name == "PaperSearchAgent"
