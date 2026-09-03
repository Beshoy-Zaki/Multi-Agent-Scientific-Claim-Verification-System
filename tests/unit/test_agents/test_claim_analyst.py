"""Unit tests for ClaimAnalystAgent."""

import pytest
from mascv.agents.claim_analyst import ClaimAnalystAgent


def test_claim_analyst_initialization():
    agent = ClaimAnalystAgent()
    assert agent.name == "ClaimAnalystAgent"
