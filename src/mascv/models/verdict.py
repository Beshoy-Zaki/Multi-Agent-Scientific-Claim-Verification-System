"""Verdict schemas and scientific report representation."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from mascv.models.claim import Claim
from mascv.models.argument import AdversarialDebate
from mascv.models.evidence import EvidenceBundle


class VerdictType(str, Enum):
    """Final assessment verdict classification."""
    SUPPORTED = "Supported"
    PARTIALLY_SUPPORTED = "Partially Supported"
    UNSUPPORTED = "Unsupported"
    INCONCLUSIVE = "Inconclusive"


class CriticFinding(BaseModel):
    """Critical validation finding from the Critic agent."""
    citation_valid: bool
    reasoning_sound: bool
    overgeneralization_detected: bool
    fair_comparison: bool
    critique_notes: str


class Verdict(BaseModel):
    """Final verdict and synthesis for an investigated claim."""
    claim_id: str
    verdict: VerdictType
    confidence: float
    critic_finding: CriticFinding
    strongest_supporting_argument: str
    strongest_counterargument: str
    unresolved_questions: List[str] = Field(default_factory=list)
    synthesis_summary: str


class ScientificReport(BaseModel):
    """Complete scientific claim verification report for a paper."""
    paper_id: str
    paper_title: str
    claims: List[Claim]
    evidence_bundles: List[EvidenceBundle]
    debates: List[AdversarialDebate]
    verdicts: List[Verdict]
    executive_summary: str
