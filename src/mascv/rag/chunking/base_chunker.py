"""Abstract chunker interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from mascv.models.paper import ResearchPaper


class BaseChunker(ABC):
    """Base interface for splitting papers into searchable chunks."""

    @abstractmethod
    def chunk(self, paper: ResearchPaper) -> List[Dict[str, Any]]:
        """Partition paper into candidate evidence chunks."""
        raise NotImplementedError
