"""Pydantic schemas for API request and response payloads."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    paper_id: str
    filename: str
    title: str = ""
    sections_count: int = 0
    message: str


class ClaimItemResponse(BaseModel):
    claim_id: str
    statement: str
    subject: str = ""
    claim_type: str = "performance"
    benchmarks: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)
    comparisons: List[str] = Field(default_factory=list)
    status: str = "extracted"
    verdict: Optional[str] = None
    confidence: Optional[float] = None
    is_finalized: bool = False


class ArgumentResponse(BaseModel):
    agent_name: str
    stance: str
    strength: str
    conclusion: str
    premises: List[str] = Field(default_factory=list)
    cited_evidence_ids: List[str] = Field(default_factory=list)
    identified_limitations: List[str] = Field(default_factory=list)


class DebateResponse(BaseModel):
    claim_id: str
    statement: str
    support_argument: Optional[ArgumentResponse] = None
    attack_argument: Optional[ArgumentResponse] = None
    evidence_count: int = 0


class CriticFindingResponse(BaseModel):
    citation_valid: bool
    reasoning_sound: bool
    overgeneralization_detected: bool
    fair_comparison: bool
    critique_notes: str


class VerdictResponse(BaseModel):
    claim_id: str
    verdict: str
    confidence: float
    critic_finding: Optional[CriticFindingResponse] = None
    strongest_supporting_argument: str = ""
    strongest_counterargument: str = ""
    synthesis_summary: str = ""


class FullReportResponse(BaseModel):
    paper_id: str
    paper_title: str
    total_claims: int
    verdicts: List[VerdictResponse] = Field(default_factory=list)
    executive_summary: str = ""
    status_message: str = ""

