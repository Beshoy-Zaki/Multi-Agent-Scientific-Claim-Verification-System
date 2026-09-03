"""Integration test for iterative feedback loop between Critic and Supervisor."""

import pytest
from mascv.core.state import InvestigationState


def test_investigation_state_init():
    state = InvestigationState()
    assert state.system_iteration == 0
    assert state.max_iterations == 3
