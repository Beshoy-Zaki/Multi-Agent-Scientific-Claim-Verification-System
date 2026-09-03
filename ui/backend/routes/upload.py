"""Paper upload and document parsing routes."""

from fastapi import APIRouter, UploadFile, File
from ui.backend.schemas import UploadResponse

router = APIRouter()


@router.post("/paper", response_model=UploadResponse)
async def upload_paper(file: UploadFile = File(...)):
    """Upload a research paper PDF for claim extraction and verification."""
    return UploadResponse(paper_id="stub-id", filename=file.filename, message="Paper uploaded successfully.")
