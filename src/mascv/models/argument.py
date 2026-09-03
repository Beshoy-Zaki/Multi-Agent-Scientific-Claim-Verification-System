"""Adversarial argument schemas for Support, Attack, and Critic agents."""

from typing import List
from pydantic import BaseModel, Field


class Argument(BaseModel):
    """An evidence-grounded argument formed by Support or Attack agent."""
    agent_name: str
    claim_id: str
    stance: str  # "FOR" or "AGAINST"
    premises: List[str] = Field(default_factory=list)
    cited_evidence_ids: List[str] = Field(default_factory=list)
    conclusion: str
    strength: str  # "Strong", "Moderate", "Weak"
    identified_limitations: List[str] = Field(default_factory=list)


class AdversarialDebate(BaseModel):
    """Container holding the dialectic debate surrounding a single claim."""
    claim_id: str
    support_argument: Argument
    attack_argument: Argument
