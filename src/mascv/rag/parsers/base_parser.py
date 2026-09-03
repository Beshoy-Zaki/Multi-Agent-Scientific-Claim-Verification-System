"""Abstract base parser for academic documents."""

from abc import ABC, abstractmethod
from mascv.models.paper import ResearchPaper


class BaseDocumentParser(ABC):
    """Abstract parser interface for converting raw files into ResearchPaper objects."""

    @abstractmethod
    def parse(self, file_path: str) -> ResearchPaper:
        """Parse document into structured ResearchPaper model."""
        raise NotImplementedError
