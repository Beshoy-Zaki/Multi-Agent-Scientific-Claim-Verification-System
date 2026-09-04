"""Supervisor Agent: Manages overall investigation and controls workflow state."""

from typing import Any, Dict, Optional
from mascv.agents.base import BaseAgent
from mascv.core.state import InvestigationState
from mascv.models.claim import ClaimStatus


class SupervisorAgent(BaseAgent):
    """Orchestrates investigation iterations and decides whether additional cycles are needed."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="SupervisorAgent", config=config)
        params = self.config.get("agent", {}).get("parameters", {}) if self.config else {}
        self.max_search_cycles = params.get("max_search_cycles", 3)
        self.min_confidence = params.get("min_confidence_to_finalize", 0.70)
        self.min_independent_sources = params.get("min_independent_sources", 2)
        self.require_replication_check = params.get("require_replication_check", True)

    def execute(self, state: InvestigationState) -> InvestigationState:
        """Evaluate claim states and dispatch next workflow action."""
        state.system_iteration += 1
        next_step = self.decide_next_step(state)
        state.metadata["last_supervisor_decision"] = next_step
        state.metadata["current_iteration"] = state.system_iteration
        return state

    def decide_next_step(self, state: InvestigationState) -> str:
        """Determine what agent should run next based on the state notebook."""
        # Rule 1: If no claims have been extracted yet, deploy Claim Analyst!
        if not state.claims:
            if state.paper or "raw_paper_text" in state.metadata:
                return "claim_analyst"
            return "end"

        # Rule 2: Safety Circuit Breaker (prevent infinite loops)
        if state.system_iteration >= state.max_iterations:
            return "finalize"

        # Rule 3: Check each claim to see what it needs
        for claim_id, claim_state in state.claims.items():
            if claim_state.is_finalized:
                continue

            # Check 1: Do we have enough independent sources?
            if len(claim_state.external_papers_found) < self.min_independent_sources:
                if claim_state.iteration_count < self.max_search_cycles:
                    state.active_claim_id = claim_id
                    claim_state.iteration_count += 1
                    return "paper_search"

            # Check 2: If a verdict was issued, is the confidence high enough?
            if claim_state.verdict and claim_state.verdict.confidence < self.min_confidence:
                if claim_state.iteration_count < self.max_search_cycles:
                    state.active_claim_id = claim_id
                    claim_state.iteration_count += 1
                    return "paper_search"

        # Rule 4: All claims have been searched and verified!
        return "finalize"
