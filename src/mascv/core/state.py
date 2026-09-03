"""Investigation state representations across the multi-agent graph."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from mascv.models.paper import ResearchPaper
from mascv.models.claim import Claim
from mascv.models.evidence import EvidenceBundle
from mascv.models.argument import Argument
from mascv.models.verdict import Verdict, CriticFinding


class ClaimInvestigationState(BaseModel):
    """State of an individual claim being investigated."""
    claim: Claim
    iteration_count: int = 0
    external_papers_found: List[str] = Field(default_factory=list)
    evidence_bundle_ids: List[str] = Field(default_factory=list)
    support_argument: Optional[Argument] = None
    attack_argument: Optional[Argument] = None
    critic_finding: Optional[CriticFinding] = None
    verdict: Optional[Verdict] = None
    is_finalized: bool = False
    status_message: str = "Initialized"


class InvestigationState(BaseModel):
    """Global system state passed across all agents in the MASCV graph."""
    paper: Optional[ResearchPaper] = None
    claims: Dict[str, ClaimInvestigationState] = Field(default_factory=dict)
    global_evidence_store: Dict[str, EvidenceBundle] = Field(default_factory=dict)
    active_claim_id: Optional[str] = None
    system_iteration: int = 0
    max_iterations: int = 3
    is_completed: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
