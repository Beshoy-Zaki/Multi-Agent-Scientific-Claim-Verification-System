"""Semantic Scholar Graph API client using direct HTTP requests."""

import logging
import os
from typing import List, Optional

import requests

from mascv.models.paper import PaperMetadata
from mascv.search.base_search import BaseSearchClient

logger = logging.getLogger(__name__)


class SemanticScholarClient(BaseSearchClient):
    """Client for retrieving peer-reviewed literature and citation graphs from Semantic Scholar."""

    API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialize Semantic Scholar client (API key is optional)."""
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")

    def search(self, query: str, max_results: int = 5) -> List[PaperMetadata]:
        """Query Semantic Scholar Graph API without requiring an API key."""
        logger.info("Querying Semantic Scholar for: '%s' (limit=%d)", query, max_results)
        papers: List[PaperMetadata] = []

        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,authors,abstract,year,venue,externalIds,url",
        }
        headers = {
            "User-Agent": "MASCV-ResearchAgent/1.0",
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key

        try:
            response = requests.get(
                self.API_URL, params=params, headers=headers, timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                for item in data.get("data", []):
                    authors = [
                        author.get("name", "")
                        for author in item.get("authors", [])
                        if author.get("name")
                    ]
                    ext_ids = item.get("externalIds") or {}
                    metadata = PaperMetadata(
                        title=item.get("title", "Untitled"),
                        authors=authors,
                        abstract=item.get("abstract"),
                        arxiv_id=ext_ids.get("ArXiv"),
                        doi=ext_ids.get("DOI"),
                        url=item.get("url"),
                        year=item.get("year"),
                        venue=item.get("venue") or "Semantic Scholar",
                    )
                    papers.append(metadata)
            else:
                logger.warning(
                    "Semantic Scholar returned status %d: %s",
                    response.status_code,
                    response.text[:200],
                )
        except Exception as e:
            logger.error("Error querying Semantic Scholar: %s", e)

        return papers
