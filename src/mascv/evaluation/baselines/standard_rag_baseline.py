"""Baseline 2: Standard fixed-chunking RAG pipeline."""

from typing import Dict, Any


class StandardRAGBaseline:
    """Traditional RAG pipeline using naive fixed-size token chunking."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def verify_claim(self, claim_text: str, corpus_path: str) -> Dict[str, Any]:
        """Standard chunking and similarity search without claim-aware bundling."""
        raise NotImplementedError("Standard RAG baseline to be implemented.")
