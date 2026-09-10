"""Compatibility shim re-exporting tools for agents."""

from mascv.tools.evidence_tools import search_scientific_evidence, calculate

__all__ = [
    "search_scientific_evidence",
    "calculate",
]
