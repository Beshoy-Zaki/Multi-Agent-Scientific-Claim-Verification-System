"""Contradiction detection evaluation metric."""

from typing import List, Dict, Any


class ContradictionDetectionMetric:
    """Quantifies success in detecting conflicting literature and non-replications."""

    def compute(self, detected_contradictions: List[Dict[str, Any]], ground_truth_contradictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute contradiction precision and recall."""
        raise NotImplementedError
