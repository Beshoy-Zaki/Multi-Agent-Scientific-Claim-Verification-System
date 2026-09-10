"""Evidence schemas, relationships, and bundles."""

from enum import Enum
from typing import Mapping, Optional, Dict, Any

from pydantic import BaseModel, Field, StrictBool


class EvidenceRelationship(str, Enum):
    """Relationship between evidence and a scientific claim."""

    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    QUALIFIES = "QUALIFIES"
    REPLICATES = "REPLICATES"
    CHALLENGES = "CHALLENGES"
    ALTERNATIVE = "ALTERNATIVE"


class SourceType(str, Enum):
    """Provenance origin of an evidence source."""

    TARGET_PAPER = "TARGET_PAPER"
    EXTERNAL_SOURCE = "EXTERNAL_SOURCE"


class EvidenceBundle(BaseModel):
    """Claim-aware evidence bundle with provenance tracking."""

    id: str

    claim_id: str

    source_paper_id: str

    source_title: str

    source_type: SourceType = SourceType.TARGET_PAPER

    is_independent: StrictBool = False

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


def is_independent_external_evidence(
    evidence: EvidenceBundle | Mapping[str, Any],
    claim_id: Optional[str] = None,
) -> bool:
    """Return true only for typed, claim-scoped independent external evidence.

    Unknown or legacy provenance deliberately fails closed: it must be upgraded
    by the retrieval path before it can influence an independence judgement.
    """
    if isinstance(evidence, Mapping):
        source_type = evidence.get("source_type")
        independent = evidence.get("is_independent")
        evidence_claim_id = evidence.get("claim_id")
    else:
        source_type = evidence.source_type
        independent = evidence.is_independent
        evidence_claim_id = evidence.claim_id

    source_type = getattr(source_type, "value", source_type)
    return (
        source_type == SourceType.EXTERNAL_SOURCE.value
        and independent is True
        and (claim_id is None or evidence_claim_id == claim_id)
    )
