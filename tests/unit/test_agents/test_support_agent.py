"""Unit tests for SupportAgent."""

import pytest
from mascv.agents.support_agent import SupportAgent


def test_support_agent_initialization():
    agent = SupportAgent()
    assert agent.name == "SupportAgent"
