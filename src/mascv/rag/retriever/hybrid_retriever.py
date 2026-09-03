"""Hybrid dense-sparse retriever combining BM25 and vector similarity."""

from typing import List, Dict, Any


class HybridRetriever:
    """Retrieves candidate chunks using hybrid dense embeddings and sparse keyword matching."""

    def retrieve(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        """Execute hybrid search across the evidence corpus."""
        raise NotImplementedError("Hybrid retrieval to be implemented.")
