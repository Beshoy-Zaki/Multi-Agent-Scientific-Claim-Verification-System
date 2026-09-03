"""Data models and schemas for the MASCV system."""

from mascv.models.claim import Claim, ClaimType
from mascv.models.evidence import EvidenceBundle, EvidenceRelationship
from mascv.models.argument import Argument, AdversarialDebate
from mascv.models.verdict import Verdict, VerdictType, ScientificReport
from mascv.models.paper import PaperMetadata, ResearchPaper, DocumentSection

__all__ = [
    "Claim",
    "ClaimType",
    "EvidenceBundle",
    "EvidenceRelationship",
    "Argument",
    "AdversarialDebate",
    "Verdict",
    "VerdictType",
    "ScientificReport",
    "PaperMetadata",
    "ResearchPaper",
    "DocumentSection",
]
