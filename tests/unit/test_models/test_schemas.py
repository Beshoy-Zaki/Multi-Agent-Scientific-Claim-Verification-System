"""Unit tests for Pydantic data schemas."""

import pytest
from pydantic import ValidationError
from mascv.models.claim import Claim, ClaimType


def test_claim_model_creation(sample_claim):
    assert sample_claim.id == "C1"
    assert sample_claim.claim_type == ClaimType.PERFORMANCE


def test_argument_requires_real_boolean_for_independent_evidence():
    from mascv.models.argument import Argument
    assert Argument(has_independent_evidence=True).has_independent_evidence is True
    assert Argument(has_independent_evidence=False).has_independent_evidence is False

    for malformed_value in ("", "true", "false", None, "hello"):
        with pytest.raises(ValidationError):
            Argument(has_independent_evidence=malformed_value)


def test_evidence_bundle_requires_real_boolean_for_independence():
    from mascv.models.evidence import EvidenceBundle, EvidenceRelationship
    bundle = EvidenceBundle(
        id="E1",
        claim_id="C1",
        source_paper_id="P1",
        source_title="Title",
        location="p1",
        content="Content",
        relationship=EvidenceRelationship.SUPPORTS,
        is_independent=True,
    )
    assert bundle.is_independent is True

    with pytest.raises(ValidationError):
        EvidenceBundle(
            id="E2", claim_id="C1", source_paper_id="P1", source_title="Title",
            location="p1", content="Content", relationship=EvidenceRelationship.SUPPORTS,
            is_independent="",
        )
