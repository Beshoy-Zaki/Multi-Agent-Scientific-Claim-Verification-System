"""Crossref DOI metadata search client."""

from typing import List
from mascv.search.base_search import BaseSearchClient
from mascv.models.paper import PaperMetadata


class CrossrefClient(BaseSearchClient):
    """Client for Crossref API metadata lookup."""

    def search(self, query: str, max_results: int = 5) -> List[PaperMetadata]:
        """Query Crossref API."""
        raise NotImplementedError("Crossref client to be implemented.")
