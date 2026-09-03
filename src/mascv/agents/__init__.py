"""Specialized agents for the MASCV system."""

from mascv.agents.base import BaseAgent
from mascv.agents.supervisor import SupervisorAgent
from mascv.agents.claim_analyst import ClaimAnalystAgent
from mascv.agents.paper_search import PaperSearchAgent
from mascv.agents.evidence_rag import EvidenceRAGAgent
from mascv.agents.support_agent import SupportAgent
from mascv.agents.attack_agent import AttackAgent
from mascv.agents.critic_agent import CriticAgent

__all__ = [
    "BaseAgent",
    "SupervisorAgent",
    "ClaimAnalystAgent",
    "PaperSearchAgent",
    "EvidenceRAGAgent",
    "SupportAgent",
    "AttackAgent",
    "CriticAgent",
]
