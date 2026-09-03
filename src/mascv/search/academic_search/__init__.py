"""Academic database search clients."""

from mascv.search.academic_search.arxiv_client import ArxivClient
from mascv.search.academic_search.semantic_scholar_client import SemanticScholarClient
from mascv.search.academic_search.crossref_client import CrossrefClient

__all__ = ["ArxivClient", "SemanticScholarClient", "CrossrefClient"]
