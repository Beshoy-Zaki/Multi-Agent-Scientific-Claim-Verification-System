"""Paper upload and document parsing routes."""

import os
import shutil
import tempfile
from uuid import uuid4
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from mascv.core.state import InvestigationState
from mascv.rag.parsers.pdf_parser import PDFParser
from ui.backend.schemas import UploadResponse
from ui.backend.session_manager import set_session

router = APIRouter()


@router.post("/paper", response_model=UploadResponse)
async def upload_paper(file: UploadFile = File(...)):
    """Upload a research paper PDF for claim extraction and verification."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        parser = PDFParser()
        paper = parser.parse(tmp_path)
        paper_id = f"P-{uuid4().hex[:8]}"
        state = InvestigationState(paper=paper)
        set_session(paper_id, state)

        title = paper.metadata.title if paper.metadata and paper.metadata.title else file.filename
        return UploadResponse(
            paper_id=paper_id,
            filename=file.filename,
            title=title,
            sections_count=len(paper.sections),
            message=f"Paper '{title}' parsed into {len(paper.sections)} sections successfully.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to parse paper: {exc}")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@router.post("/sample", response_model=UploadResponse)
async def load_sample_paper(sample_name: str = Query(default="lora_2106.09685.pdf")):
    """Load a packaged sample paper directly from data/sample_inputs/."""
    sample_path = os.path.join("data", "sample_inputs", sample_name)
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail=f"Sample paper '{sample_name}' not found.")

    try:
        parser = PDFParser()
        paper = parser.parse(sample_path)
        paper_id = f"P-{uuid4().hex[:8]}"
        state = InvestigationState(paper=paper)
        set_session(paper_id, state)

        title = paper.metadata.title if paper.metadata and paper.metadata.title else sample_name
        return UploadResponse(
            paper_id=paper_id,
            filename=sample_name,
            title=title,
            sections_count=len(paper.sections),
            message=f"Sample paper '{title}' parsed into {len(paper.sections)} sections successfully.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load sample paper: {exc}")

