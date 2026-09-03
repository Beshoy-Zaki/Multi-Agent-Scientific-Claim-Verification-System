"""Integration tests for end-to-end verification pipeline."""

import pytest
from mascv.core.workflow import MASCVWorkflow


def test_workflow_init():
    wf = MASCVWorkflow()
    assert wf is not None
