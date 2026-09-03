"""Unit tests for SupervisorAgent."""

import pytest
from mascv.agents.supervisor import SupervisorAgent


def test_supervisor_initialization():
    agent = SupervisorAgent()
    assert agent.name == "SupervisorAgent"
