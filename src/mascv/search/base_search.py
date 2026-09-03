"""Abstract search client interface."""

from abc import ABC, abstractmethod
from typing import List
from mascv.models.paper import PaperMetadata


class BaseSearchClient(ABC):
    """Abstract search engine client for discovering external scientific papers."""

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> List[PaperMetadata]:
        """Query literature provider and return paper metadata."""
        raise NotImplementedError
