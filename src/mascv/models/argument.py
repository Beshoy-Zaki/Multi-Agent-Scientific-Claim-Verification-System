"""Adversarial argument schemas for Support, Attack, and Critic agents."""

from typing import List
from pydantic import BaseModel, Field, StrictBool


class Argument(BaseModel):
    """An evidence-grounded argument formed by Support or Attack agent."""
    agent_name: str = Field(default="SupportAgent")
    claim_id: str = Field(default="")
    stance: str = Field(default="FOR")  # "FOR" or "AGAINST"
    premises: List[str] = Field(default_factory=list)
    cited_evidence_ids: List[str] = Field(default_factory=list)
    conclusion: str = Field(default="")
    strength: str = Field(default="Moderate")  # "Strong", "Moderate", "Weak"
    identified_limitations: List[str] = Field(default_factory=list)
    # This is application-derived provenance, never an LLM judgement.  A
    # missing value must fail closed rather than turn an unverified argument
    # into independent support.
    has_independent_evidence: StrictBool = Field(default=False)
    evidence_types_used: List[str] = Field(default_factory=list)


class AdversarialDebate(BaseModel):
    """Container holding the dialectic debate surrounding a single claim."""
    claim_id: str
    support_argument: Argument
    attack_argument: Argument
