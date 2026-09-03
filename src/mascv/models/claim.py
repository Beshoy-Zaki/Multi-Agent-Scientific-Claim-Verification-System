"""Scientific claim definitions and categorization schemas."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class ClaimType(str, Enum):
    """Taxonomy of scientific claims evaluated by the system."""
    PERFORMANCE = "performance"
    EFFICIENCY = "efficiency"
    CAUSAL = "causal"
    GENERALIZATION = "generalization"
    METHODOLOGICAL = "methodological"
    NOVELTY = "novelty"


class ClaimStatus(str, Enum):
    """Investigation lifecycle status for a claim."""
    EXTRACTED = "extracted"
    UNDER_INVESTIGATION = "under_investigation"
    NEEDS_MORE_EVIDENCE = "needs_more_evidence"
    VERIFIED = "verified"


class Claim(BaseModel):
    """Structured proposition extracted from a scientific paper."""
    id: str
    paper_id: str
    subject: str
    statement: str
    claim_type: ClaimType
    benchmarks: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)
    comparisons: List[str] = Field(default_factory=list)
    conditions: Optional[str] = None
    status: ClaimStatus = ClaimStatus.EXTRACTED
    source_location: Optional[str] = None
