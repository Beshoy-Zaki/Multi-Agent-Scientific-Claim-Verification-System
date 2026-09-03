"""Shared test fixtures and environment configuration for pytest."""

import pytest
from mascv.models.claim import Claim, ClaimType
from mascv.models.evidence import EvidenceBundle, EvidenceRelationship
from mascv.models.argument import Argument
from mascv.models.paper import ResearchPaper, PaperMetadata


@pytest.fixture
def sample_claim():
    return Claim(
        id="C1",
        paper_id="paper-001",
        subject="Method X",
        statement="Method X improves accuracy on Benchmark A.",
        claim_type=ClaimType.PERFORMANCE,
        benchmarks=["Benchmark A"],
        metrics=["Accuracy"],
        comparisons=["Baseline Y"],
        conditions="Standard training parameters",
    )


@pytest.fixture
def sample_evidence_bundle():
    return EvidenceBundle(
        id="E1",
        claim_id="C1",
        source_paper_id="ext-paper-101",
        source_title="Replication Study on Method X",
        location="Page 5, Table 2",
        content="Method X achieved 82.5% vs Baseline Y 78.1% on Benchmark A.",
        relationship=EvidenceRelationship.SUPPORTS,
        confidence_score=0.92,
    )
