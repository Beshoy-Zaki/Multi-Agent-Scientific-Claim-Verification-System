"""Adversarial debate (Support vs Attack) routes."""

from fastapi import APIRouter, HTTPException
from mascv.agents.paper_search import PaperSearchAgent
from mascv.agents.evidence_rag import EvidenceRAGAgent
from mascv.agents.support_agent import SupportAgent
from mascv.agents.attack_agent import AttackAgent
from ui.backend.schemas import DebateResponse, ArgumentResponse
from ui.backend.session_manager import get_session

router = APIRouter()


def _format_argument(arg) -> ArgumentResponse:
    if not arg:
        return None
    agent_name = getattr(arg, "agent_name", arg.get("agent_name", "UnknownAgent") if isinstance(arg, dict) else "UnknownAgent")
    stance = getattr(arg, "stance", arg.get("stance", "FOR") if isinstance(arg, dict) else "FOR")
    strength = getattr(arg, "strength", arg.get("strength", "Moderate") if isinstance(arg, dict) else "Moderate")
    conclusion = getattr(arg, "conclusion", arg.get("conclusion", "") if isinstance(arg, dict) else "")
    premises = getattr(arg, "premises", arg.get("premises", []) if isinstance(arg, dict) else [])
    cited_ids = getattr(arg, "cited_evidence_ids", arg.get("cited_evidence_ids", []) if isinstance(arg, dict) else [])
    limitations = getattr(arg, "identified_limitations", arg.get("identified_limitations", []) if isinstance(arg, dict) else [])

    return ArgumentResponse(
        agent_name=str(agent_name),
        stance=str(stance),
        strength=str(strength),
        conclusion=str(conclusion),
        premises=list(premises),
        cited_evidence_ids=list(cited_ids),
        identified_limitations=list(limitations),
    )


def _get_debate_response(state, claim_id: str) -> DebateResponse:
    claims = state.claims if hasattr(state, "claims") else state.get("claims", {})
    if claim_id not in claims:
        raise HTTPException(status_code=404, detail=f"Claim '{claim_id}' not found.")

    cstate = claims[claim_id]
    claim = cstate.get("claim") if isinstance(cstate, dict) else getattr(cstate, "claim", None)
    statement = claim.get("statement", "") if isinstance(claim, dict) else getattr(claim, "statement", "")

    support_arg = cstate.get("support_argument") if isinstance(cstate, dict) else getattr(cstate, "support_argument", None)
    attack_arg = cstate.get("attack_argument") if isinstance(cstate, dict) else getattr(cstate, "attack_argument", None)
    evidence_ids = cstate.get("evidence_bundle_ids", []) if isinstance(cstate, dict) else getattr(cstate, "evidence_bundle_ids", [])

    return DebateResponse(
        claim_id=claim_id,
        statement=statement,
        support_argument=_format_argument(support_arg),
        attack_argument=_format_argument(attack_arg),
        evidence_count=len(evidence_ids),
    )


@router.post("/investigate/{paper_id}/{claim_id}", response_model=DebateResponse)
async def investigate_claim(paper_id: str, claim_id: str):
    """Execute PaperSearch, EvidenceRAG, SupportAgent, and AttackAgent for a target claim."""
    state = get_session(paper_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Paper session '{paper_id}' not found.")

    claims = state.claims if hasattr(state, "claims") else state.get("claims", {})
    if claim_id not in claims:
        raise HTTPException(status_code=404, detail=f"Claim '{claim_id}' not found in paper state.")

    state.active_claim_id = claim_id
    try:
        # 1. Search for external literature if needed
        search_agent = PaperSearchAgent()
        state = search_agent.execute(state)

        # 2. Extract and bundle claim-aware evidence
        rag_agent = EvidenceRAGAgent()
        state = rag_agent.execute(state)

        # 3. Construct proponent affirmative case
        support_agent = SupportAgent()
        state = support_agent.execute(state)

        # 4. Construct adversarial counterargument
        attack_agent = AttackAgent()
        state = attack_agent.execute(state)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Debate investigation failed: {exc}")

    return _get_debate_response(state, claim_id)


@router.get("/{paper_id}/{claim_id}", response_model=DebateResponse)
async def get_adversarial_debate(paper_id: str, claim_id: str):
    """Retrieve the proponent and opponent arguments for a claim."""
    state = get_session(paper_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Paper session '{paper_id}' not found.")

    return _get_debate_response(state, claim_id)

