"""Unit tests for AttackAgent."""

import pytest
from mascv.agents.attack_agent import AttackAgent


def test_attack_agent_initialization():
    agent = AttackAgent()
    assert agent.name == "AttackAgent"
