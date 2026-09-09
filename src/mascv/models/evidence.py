"""Evidence schemas, relationships, and bundles."""

from enum import Enum
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field


class EvidenceRelationship(str, Enum):
    """Relationship between evidence and a scientific claim."""

    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    QUALIFIES = "QUALIFIES"
    REPLICATES = "REPLICATES"
    CHALLENGES = "CHALLENGES"
    ALTERNATIVE = "ALTERNATIVE"


class EvidenceBundle(BaseModel):
    """Claim-aware evidence bundle with provenance tracking."""

    id: str

    claim_id: str

    source_paper_id: str

    source_title: str

    location: str

    content: str

    context: Optional[str] = None

    relationship: EvidenceRelationship

    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    experimental_conditions: Optional[
        Dict[str, Any]
    ] = None

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            val = getattr(self, key)
            if isinstance(val, Enum):
                return val.value
            return val
        raise KeyError(key)