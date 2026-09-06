"""arXiv API search client using direct Python library."""

import logging
from typing import List

import arxiv

from mascv.models.paper import PaperMetadata
from mascv.search.base_search import BaseSearchClient

logger = logging.getLogger(__name__)


class ArxivClient(BaseSearchClient):
    """Client for retrieving preprints from arXiv without requiring an API key."""

    def __init__(self) -> None:
        """Initialize the arXiv client."""
        self.client = arxiv.Client(
            page_size=10,
            delay_seconds=1.0,
            num_retries=3,
        )

    def search(self, query: str, max_results: int = 5) -> List[PaperMetadata]:
        """Query arXiv API and parse results into standard PaperMetadata objects."""
        logger.info("Querying arXiv for: '%s' (max_results=%d)", query, max_results)
        papers: List[PaperMetadata] = []

        try:
            search_query = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )

            for result in self.client.results(search_query):
                clean_abstract = (
                    result.summary.replace("\n", " ").strip() if result.summary else None
                )
                published_year = result.published.year if result.published else None

                metadata = PaperMetadata(
                    title=result.title.replace("\n", " ").strip(),
                    authors=[author.name for author in result.authors],
                    abstract=clean_abstract,
                    arxiv_id=result.get_short_id(),
                    doi=result.doi,
                    url=result.entry_id,
                    year=published_year,
                    venue="arXiv",
                )
                papers.append(metadata)

        except Exception as e:
            logger.error("Error fetching results from arXiv for query '%s': %s", query, e)

        return papers
