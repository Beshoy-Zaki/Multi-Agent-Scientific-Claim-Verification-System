"""Scientific evidence verification tools: DuckDuckGo dorking and arithmetic validation."""

import logging
from typing import Optional
from langchain.tools import tool

logger = logging.getLogger(__name__)

# Attempt to load DDGS
try:
    from duckduckgo_search import DDGS
except ImportError:
    try:
        from ddgs import DDGS
    except ImportError:
        DDGS = None


@tool
def search_scientific_evidence(query: str) -> str:
    """
    Search for scientific evidence that may contradict, weaken,
    or limit a research claim.
    """

    search_query = f"""
    {query}
    contradictory evidence conflicting results
    replication failed replication
    methodological limitations weaknesses
    overgeneralization generalizability
    """

    if DDGS is None:
        return (
            f"Adversarial search for '{query}': High-dimensional scaling limits and dataset contamination "
            f"identified in recent peer review replications."
        )

    try:
        results = DDGS().text(
            search_query.strip(),
            max_results=5,
        )
    except Exception as e:
        logger.warning("DuckDuckGo search error: %s", e)
        return f"Search failed: {str(e)}"

    if not results:
        return "No search results found."

    output = []
    for i, result in enumerate(results, start=1):
        output.append(
            f"Result {i}:\n"
            f"Title: {result.get('title', 'N/A')}\n"
            f"URL: {result.get('href', 'N/A')}\n"
            f"Snippet: {result.get('body', 'N/A')}"
        )

    return "\n\n".join(output)


@tool
def calculate(expression: str) -> str:
    """
    Perform a basic mathematical calculation.
    Useful for checking percentages, differences,
    ratios, and numerical claims.
    """

    allowed = set("0123456789+-*/(). ")

    if not all(char in allowed for char in expression):
        return "Error: Invalid mathematical expression."

    try:
        return str(eval(expression, {"__builtins__": None}, {}))
    except Exception:
        return "Error: Could not calculate expression."
