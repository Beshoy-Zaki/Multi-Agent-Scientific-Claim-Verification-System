"""Structured repository for storing and querying evidence bundles."""

from typing import Dict, List, Optional
from mascv.models.evidence import EvidenceBundle


class EvidenceStore:
    """Shared state store housing all evidence bundles extracted across papers."""

    def __init__(self) -> None:
        self._store: Dict[str, EvidenceBundle] = {}

    def add(self, bundle: EvidenceBundle) -> None:
        """Add an evidence bundle to the repository."""
        self._store[bundle.id] = bundle

    def get_by_claim(self, claim_id: str) -> List[EvidenceBundle]:
        """Retrieve all evidence bundles linked to a specific claim."""
        return [b for b in self._store.values() if b.claim_id == claim_id]

    def get_by_relationship(self, claim_id: str, relationship: str) -> List[EvidenceBundle]:
        """Retrieve evidence for a claim matching a specific relationship."""
        return [
            b for b in self._store.values()
            if b.claim_id == claim_id and b.relationship == relationship
        ]
