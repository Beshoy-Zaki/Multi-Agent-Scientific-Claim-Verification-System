"""Scientific assessment reports export and display routes."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/{paper_id}")
async def get_full_report(paper_id: str):
    """Retrieve complete evidence-grounded scientific assessment report."""
    return {}
