"""Extracted claims inspection and management routes."""

from typing import List
from fastapi import APIRouter, HTTPException
from mascv.agents.claim_analyst import ClaimAnalystAgent
from ui.backend.schemas import ClaimItemResponse
from ui.backend.session_manager import get_session

router = APIRouter()


def _format_claim_item(cid: str, cstate) -> ClaimItemResponse:
    claim = cstate.get("claim") if isinstance(cstate, dict) else getattr(cstate, "claim", None)
    statement = claim.get("statement", "") if isinstance(claim, dict) else getattr(claim, "statement", "")
    subject = claim.get("subject", "") if isinstance(claim, dict) else getattr(claim, "subject", "")
    claim_type = claim.get("claim_type", "performance") if isinstance(claim, dict) else getattr(claim, "claim_type", "performance")
    if hasattr(claim_type, "value"):
        claim_type = claim_type.value
    benchmarks = claim.get("benchmarks", []) if isinstance(claim, dict) else getattr(claim, "benchmarks", [])
    metrics = claim.get("metrics", []) if isinstance(claim, dict) else getattr(claim, "metrics", [])
    comparisons = claim.get("comparisons", []) if isinstance(claim, dict) else getattr(claim, "comparisons", [])

    verdict_obj = cstate.get("verdict") if isinstance(cstate, dict) else getattr(cstate, "verdict", None)
    v_str = None
    v_conf = None
    if verdict_obj:
        v_val = getattr(verdict_obj, "verdict", verdict_obj.get("verdict") if isinstance(verdict_obj, dict) else None)
        v_str = getattr(v_val, "value", str(v_val)) if v_val else None
        v_conf = getattr(verdict_obj, "confidence", verdict_obj.get("confidence") if isinstance(verdict_obj, dict) else None)

    is_final = cstate.get("is_finalized", False) if isinstance(cstate, dict) else getattr(cstate, "is_finalized", False)
    status = "finalized" if is_final else ("verified" if verdict_obj else "extracted")

    return ClaimItemResponse(
        claim_id=cid,
        statement=statement,
        subject=subject,
        claim_type=str(claim_type),
        benchmarks=list(benchmarks),
        metrics=list(metrics),
        comparisons=list(comparisons),
        status=status,
        verdict=v_str,
        confidence=v_conf,
        is_finalized=is_final,
    )


@router.post("/extract/{paper_id}", response_model=List[ClaimItemResponse])
async def extract_claims(paper_id: str):
    """Extract testable scientific claims from the ingested paper using ClaimAnalystAgent."""
    state = get_session(paper_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Paper session '{paper_id}' not found.")

    if not state.claims:
        try:
            analyst = ClaimAnalystAgent()
            state = analyst.execute(state)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Claim extraction failed: {exc}")

    return [_format_claim_item(cid, cstate) for cid, cstate in state.claims.items()]


@router.get("/{paper_id}", response_model=List[ClaimItemResponse])
async def get_claims(paper_id: str):
    """Retrieve all extracted claims for an ingested paper."""
    state = get_session(paper_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Paper session '{paper_id}' not found.")

    return [_format_claim_item(cid, cstate) for cid, cstate in state.claims.items()]
