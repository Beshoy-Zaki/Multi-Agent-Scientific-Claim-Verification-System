"""Web search client for broader scientific and technical web discovery."""

from typing import List
from mascv.search.base_search import BaseSearchClient
from mascv.models.paper import PaperMetadata


class WebSearchClient(BaseSearchClient):
    """Client for general search engines (e.g., Tavily, Serper)."""

    def search(self, query: str, max_results: int = 5) -> List[PaperMetadata]:
        """Query web search provider."""
        raise NotImplementedError("Web search client to be implemented.")
