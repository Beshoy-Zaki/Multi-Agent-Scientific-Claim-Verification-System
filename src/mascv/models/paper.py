"""Research paper representation and metadata schemas."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DocumentSection(BaseModel):
    """Represents a structured section within a scientific paper."""
    title: str
    content: str
    page_number: Optional[int] = None
    subsections: List["DocumentSection"] = Field(default_factory=list)


class PaperMetadata(BaseModel):
    """Bibliographic metadata for a scientific publication."""
    title: str
    authors: List[str] = Field(default_factory=list)
    abstract: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    url: Optional[str] = None
    year: Optional[int] = None
    venue: Optional[str] = None


class ResearchPaper(BaseModel):
    """Represents a full research paper ingested into the MASCV system."""
    id: str
    metadata: PaperMetadata
    raw_text: str
    sections: List[DocumentSection] = Field(default_factory=list)
    tables_and_figures: List[Dict[str, Any]] = Field(default_factory=list)
