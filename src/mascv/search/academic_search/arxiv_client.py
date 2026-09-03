"""arXiv API search client."""

from typing import List
from mascv.search.base_search import BaseSearchClient
from mascv.models.paper import PaperMetadata


class ArxivClient(BaseSearchClient):
    """Client for retrieving preprints from arXiv."""

    def search(self, query: str, max_results: int = 5) -> List[PaperMetadata]:
        """Query arXiv API and parse results."""
        raise NotImplementedError("arXiv search client to be implemented.")
