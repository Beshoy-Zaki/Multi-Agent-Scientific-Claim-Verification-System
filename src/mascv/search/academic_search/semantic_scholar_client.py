"""Semantic Scholar Graph API client."""

from typing import List
from mascv.search.base_search import BaseSearchClient
from mascv.models.paper import PaperMetadata


class SemanticScholarClient(BaseSearchClient):
    """Client for retrieving peer-reviewed literature and citation graphs."""

    def search(self, query: str, max_results: int = 5) -> List[PaperMetadata]:
        """Query Semantic Scholar API."""
        raise NotImplementedError("Semantic Scholar client to be implemented.")
