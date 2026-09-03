"""Baseline 1: Monolithic single-agent claim verification system."""

from typing import Dict, Any


class SingleAgentBaseline:
    """Single strong LLM agent with direct search and retrieval tools."""

    def __init__(self, model_name: str = "gpt-4o") -> None:
        self.model_name = model_name

    def verify_paper(self, paper_path: str) -> Dict[str, Any]:
        """Perform end-to-end verification without adversarial multi-agent decomposition."""
        raise NotImplementedError("Single-agent baseline to be implemented.")
