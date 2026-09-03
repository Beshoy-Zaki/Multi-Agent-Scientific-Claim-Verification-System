"""Unit tests for CriticAgent."""

import pytest
from mascv.agents.critic_agent import CriticAgent


def test_critic_agent_initialization():
    agent = CriticAgent()
    assert agent.name == "CriticAgent"
