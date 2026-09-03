"""Document cache to avoid redundant fetching and parsing of papers."""

from typing import Optional
from mascv.models.paper import ResearchPaper


class DocumentCache:
    """Disk/in-memory cache for parsed research papers."""

    def __init__(self, cache_dir: str = "data/processed_cache") -> None:
        self.cache_dir = cache_dir

    def get(self, identifier: str) -> Optional[ResearchPaper]:
        """Fetch cached paper if exists."""
        raise NotImplementedError("Cache retrieval to be implemented.")

    def put(self, identifier: str, paper: ResearchPaper) -> None:
        """Save parsed paper into cache."""
        raise NotImplementedError("Cache persistence to be implemented.")
