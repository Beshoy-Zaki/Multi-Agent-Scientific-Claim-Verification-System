"""Scientific assessment reports export and display routes."""

from fastapi import APIRouter, HTTPException
from mascv.agents.critic_agent import CriticAgent
from mascv.agents.supervisor import SupervisorAgent
from ui.backend.schemas import VerdictResponse, CriticFindingResponse, FullReportResponse
from ui.backend.session_manager import get_session

router = APIRouter()


def _format_verdict(cid: str, verdict) -> VerdictResponse:
    if not verdict:
        return None
    v_type = getattr(verdict, "verdict", verdict.get("verdict") if isinstance(verdict, dict) else "Inconclusive")
    v_str = getattr(v_type, "value", str(v_type))
    conf = getattr(verdict, "confidence", verdict.get("confidence", 0.0) if isinstance(verdict, dict) else 0.0)

    finding_obj = getattr(verdict, "critic_finding", verdict.get("critic_finding") if isinstance(verdict, dict) else None)
    finding_resp = None
    if finding_obj:
        finding_resp = CriticFindingResponse(
            citation_valid=bool(getattr(finding_obj, "citation_valid", finding_obj.get("citation_valid", False) if isinstance(finding_obj, dict) else False)),
            reasoning_sound=bool(getattr(finding_obj, "reasoning_sound", finding_obj.get("reasoning_sound", False) if isinstance(finding_obj, dict) else False)),
            overgeneralization_detected=bool(getattr(finding_obj, "overgeneralization_detected", finding_obj.get("overgeneralization_detected", False) if isinstance(finding_obj, dict) else False)),
            fair_comparison=bool(getattr(finding_obj, "fair_comparison", finding_obj.get("fair_comparison", False) if isinstance(finding_obj, dict) else False)),
            critique_notes=str(getattr(finding_obj, "critique_notes", finding_obj.get("critique_notes", "") if isinstance(finding_obj, dict) else "")),
        )

    strong_sup = getattr(verdict, "strongest_supporting_argument", verdict.get("strongest_supporting_argument", "") if isinstance(verdict, dict) else "")
    strong_atk = getattr(verdict, "strongest_counterargument", verdict.get("strongest_counterargument", "") if isinstance(verdict, dict) else "")
    synth = getattr(verdict, "synthesis_summary", verdict.get("synthesis_summary", "") if isinstance(verdict, dict) else "")

    return VerdictResponse(
        claim_id=cid,
        verdict=v_str,
        confidence=float(conf),
        critic_finding=finding_resp,
        strongest_supporting_argument=strong_sup,
        strongest_counterargument=strong_atk,
        synthesis_summary=synth,
    )


@router.post("/adjudicate/{paper_id}/{claim_id}", response_model=VerdictResponse)
async def adjudicate_claim(paper_id: str, claim_id: str):
    """Execute CriticAgent to evaluate dialectic debate and synthesize a scientific verdict."""
    state = get_session(paper_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Paper session '{paper_id}' not found.")

    claims = state.claims if hasattr(state, "claims") else state.get("claims", {})
    if claim_id not in claims:
        raise HTTPException(status_code=404, detail=f"Claim '{claim_id}' not found in paper state.")

    state.active_claim_id = claim_id
    try:
        critic = CriticAgent()
        state = critic.execute(state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Adjudication failed: {exc}")

    cstate = claims[claim_id]
    verdict = cstate.get("verdict") if isinstance(cstate, dict) else getattr(cstate, "verdict", None)
    return _format_verdict(claim_id, verdict)


@router.get("/{paper_id}", response_model=FullReportResponse)
async def get_full_report(paper_id: str):
    """Retrieve complete evidence-grounded scientific assessment report with executive summary."""
    state = get_session(paper_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Paper session '{paper_id}' not found.")

    paper_title = (
        state.paper.metadata.title
        if state.paper and state.paper.metadata and state.paper.metadata.title
        else "Target Scientific Paper"
    )

    supervisor = SupervisorAgent()
    executive_summary = supervisor.generate_executive_summary(state)

    verdicts = []
    for cid, cstate in state.claims.items():
        v = cstate.get("verdict") if isinstance(cstate, dict) else getattr(cstate, "verdict", None)
        if v:
            verdicts.append(_format_verdict(cid, v))

    return FullReportResponse(
        paper_id=paper_id,
        paper_title=paper_title,
        total_claims=len(state.claims),
        verdicts=verdicts,
        executive_summary=executive_summary,
        status_message=f"Report synthesized across {len(verdicts)} verified claim(s).",
    )
