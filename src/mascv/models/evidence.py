"""Evidence schemas, relationships, and bundles."""

from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel


class EvidenceRelationship(str, Enum):
    """Relationship between an evidence snippet and a scientific claim."""
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    QUALIFIES = "QUALIFIES"
    REPLICATES = "REPLICATES"
    CHALLENGES = "CHALLENGES"
    ALTERNATIVE = "ALTERNATIVE"


class EvidenceBundle(BaseModel):
    """Claim-aware evidence bundle with full provenance tracking."""
    id: str
    claim_id: str
    source_paper_id: str
    source_title: str
    location: str
    content: str
    context: Optional[str] = None
    relationship: EvidenceRelationship
    confidence_score: float = 0.0
    experimental_conditions: Optional[Dict[str, Any]] = None
