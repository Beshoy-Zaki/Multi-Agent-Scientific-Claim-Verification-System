"""Claim verification accuracy metric."""

from typing import List, Dict, Any


class ClaimAccuracyMetric:
    """Measures alignment between predicted verdicts and ground truth expert labels."""

    def compute(self, predictions: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate accuracy, precision, recall, and F1 across verdict classes."""
        raise NotImplementedError
