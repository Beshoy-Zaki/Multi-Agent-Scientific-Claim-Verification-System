"""Abstract embedding model interface."""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingModel(ABC):
    """Abstract embedding model for generating dense vector representations."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Compute dense embeddings for input texts."""
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Compute embedding vector for a search query."""
        raise NotImplementedError
