"""Pydantic schemas for API request and response payloads."""

from typing import List, Optional
from pydantic import BaseModel


class UploadResponse(BaseModel):
    paper_id: str
    filename: str
    message: str


class ClaimItemResponse(BaseModel):
    claim_id: str
    statement: str
    claim_type: str
    verdict: Optional[str] = None
    confidence: Optional[float] = None
