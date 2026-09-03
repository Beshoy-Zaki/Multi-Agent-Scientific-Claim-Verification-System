"""Extracted claims inspection and management routes."""

from typing import List
from fastapi import APIRouter
from ui.backend.schemas import ClaimItemResponse

router = APIRouter()


@router.get("/{paper_id}", response_model=List[ClaimItemResponse])
async def get_claims(paper_id: str):
    """Retrieve all extracted claims for an ingested paper."""
    return []
