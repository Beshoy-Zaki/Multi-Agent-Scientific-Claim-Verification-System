"""Evidence retrieval quality metric."""

from typing import List, Dict, Any


class RetrievalQualityMetric:
    """Computes precision, recall, MAP, and NDCG for retrieved literature."""

    def compute(self, retrieved: List[str], relevant_ground_truth: List[str]) -> Dict[str, float]:
        """Calculate retrieval evaluation scores."""
        raise NotImplementedError
