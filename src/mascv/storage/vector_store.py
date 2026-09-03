"""Vector database manager interface."""

from typing import List, Dict, Any


class VectorStoreManager:
    """Manages dense vector indexing and nearest neighbor queries for text chunks."""

    def __init__(self, collection_name: str = "mascv_evidence") -> None:
        self.collection_name = collection_name

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Store chunk vectors and metadata."""
        raise NotImplementedError("Vector indexing to be implemented.")

    def query(self, query_vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve nearest chunks based on cosine similarity."""
        raise NotImplementedError("Vector querying to be implemented.")
