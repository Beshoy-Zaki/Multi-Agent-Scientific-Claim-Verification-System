"""Crossref DOI metadata search client using direct HTTP requests."""

import logging
from typing import List

import requests

from mascv.models.paper import PaperMetadata
from mascv.search.base_search import BaseSearchClient

logger = logging.getLogger(__name__)


class CrossrefClient(BaseSearchClient):
    """Client for Crossref API metadata lookup (no API key required)."""

    API_URL = "https://api.crossref.org/works"

    def __init__(self, mailto: str = "researcher@example.com") -> None:
        """Initialize Crossref client with polite pool user agent."""
        self.mailto = mailto

    def search(self, query: str, max_results: int = 5) -> List[PaperMetadata]:
        """Query Crossref API for authoritative publication records."""
        logger.info("Querying Crossref for: '%s' (limit=%d)", query, max_results)
        papers: List[PaperMetadata] = []

        params = {
            "query": query,
            "rows": max_results,
        }
        headers = {
            "User-Agent": f"MASCV-ClaimVerification/1.0 (mailto:{self.mailto})",
        }

        try:
            response = requests.get(
                self.API_URL, params=params, headers=headers, timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                items = data.get("message", {}).get("items", [])
                for item in items:
                    titles = item.get("title", [])
                    title = titles[0] if titles else "Untitled"

                    authors = []
                    for author in item.get("author", []):
                        given = author.get("given", "")
                        family = author.get("family", "")
                        name = f"{given} {family}".strip()
                        if name:
                            authors.append(name)

                    # Extract publication year from issued date-parts
                    year = None
                    date_parts = item.get("issued", {}).get("date-parts", [])
                    if date_parts and date_parts[0]:
                        year = date_parts[0][0]

                    container = item.get("container-title", [])
                    venue = container[0] if container else "Crossref"

                    metadata = PaperMetadata(
                        title=title,
                        authors=authors,
                        abstract=item.get("abstract"),
                        doi=item.get("DOI"),
                        url=item.get("URL"),
                        year=year,
                        venue=venue,
                    )
                    papers.append(metadata)
        except Exception as e:
            logger.error("Error querying Crossref: %s", e)

        return papers
