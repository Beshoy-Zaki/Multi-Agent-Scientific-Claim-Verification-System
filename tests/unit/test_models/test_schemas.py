"""Unit tests for Pydantic data schemas."""

import pytest
from mascv.models.claim import Claim, ClaimType


def test_claim_model_creation(sample_claim):
    assert sample_claim.id == "C1"
    assert sample_claim.claim_type == ClaimType.PERFORMANCE
