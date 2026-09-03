"""Fixed-token size chunker used as an evaluation baseline."""

from typing import List, Dict, Any
from mascv.rag.chunking.base_chunker import BaseChunker
from mascv.models.paper import ResearchPaper


class FixedSizeChunker(BaseChunker):
    """Standard fixed-window chunking (e.g. 512 tokens with 50 token overlap)."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, paper: ResearchPaper) -> List[Dict[str, Any]]:
        """Segment paper using sliding token windows."""
        raise NotImplementedError("Fixed chunking to be implemented.")
