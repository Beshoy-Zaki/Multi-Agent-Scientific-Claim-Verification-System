"""Supervisor Agent: Manages overall investigation and controls workflow state."""

from typing import Any, Dict, Optional
from mascv.agents.base import BaseAgent
from mascv.core.state import InvestigationState
from mascv.models.claim import ClaimStatus
from mascv.models.verdict import VerdictType
from mascv.utils.logger import get_logger

logger = get_logger(__name__)


class SupervisorAgent(BaseAgent):
    """Orchestrates investigation iterations and decides whether additional cycles are needed."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        llm_client: Optional[Any] = None,
    ) -> None:
        super().__init__(name="SupervisorAgent", config=config)
        self.llm_client = llm_client

        agent_config = self.config.get("agent", {}) if self.config else {}
        self.model_name = agent_config.get("model", "gemma-4-31b-it")
        params = agent_config.get("parameters", {})
        self.max_search_cycles = params.get("max_search_cycles", 3)
        self.min_confidence = params.get("min_confidence_to_finalize", 0.70)
        self.min_independent_sources = params.get("min_independent_sources", 2)
        self.require_replication_check = params.get("require_replication_check", True)

        # Load prompt templates for executive summary synthesis
        prompts = self.config.get("prompts", {}) if self.config else {}
        self.system_prompt = prompts.get("system_prompt", "")
        self.user_prompt_template = prompts.get("user_prompt", "")

    def execute(self, state: InvestigationState) -> InvestigationState:
        """Evaluate claim states, update status logs, and dispatch next workflow action."""
        state.system_iteration += 1
        next_step = self.decide_next_step(state)

        # Build informative tracking message for live UI and console logs
        active_info = f" (Active Claim: '{state.active_claim_id}')" if state.active_claim_id else ""
        status_message = (
            f"Iteration {state.system_iteration}/{state.max_iterations}: "
            f"Routing to '{next_step}'{active_info}."
        )

        logger.info(status_message)
        state.metadata["status_message"] = status_message
        state.metadata["last_supervisor_decision"] = next_step
        state.metadata["current_iteration"] = state.system_iteration

        # When all iterations complete, mark finished and synthesize executive summary
        if next_step == "finalize":
            state.is_completed = True
            if "executive_summary" not in state.metadata:
                state.metadata["executive_summary"] = self.generate_executive_summary(state)

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

    def generate_executive_summary(self, state: InvestigationState) -> str:
        """Synthesize an executive summary of all claims and verdicts using LLM or structured fallback."""
        paper_title = (
            state.paper.metadata.title
            if state.paper and state.paper.metadata and state.paper.metadata.title
            else "Target Scientific Paper"
        )
        total_claims = len(state.claims)

        # Count verdict distributions
        supported_count = sum(
            1 for c in state.claims.values()
            if c.verdict and c.verdict.verdict == VerdictType.SUPPORTED
        )
        partially_supported_count = sum(
            1 for c in state.claims.values()
            if c.verdict and c.verdict.verdict == VerdictType.PARTIALLY_SUPPORTED
        )
        unsupported_count = sum(
            1 for c in state.claims.values()
            if c.verdict and c.verdict.verdict == VerdictType.UNSUPPORTED
        )

        # Build detailed verdicts breakdown
        breakdown_lines = []
        for cid, c_state in state.claims.items():
            v_type = c_state.verdict.verdict.value if c_state.verdict else "Unverified"
            v_conf = f"{c_state.verdict.confidence:.2f}" if c_state.verdict else "N/A"
            v_summary = (
                c_state.verdict.synthesis_summary
                if c_state.verdict and c_state.verdict.synthesis_summary
                else "No synthesis summary available."
            )
            breakdown_lines.append(
                f"- Claim {cid} [{v_type}, Confidence: {v_conf}]: {c_state.claim.statement}\n"
                f"  Synthesis: {v_summary}"
            )
        verdicts_breakdown = (
            "\n".join(breakdown_lines) if breakdown_lines else "No individual claim verdicts recorded."
        )

        # If an LLM client is configured, generate executive summary via LLM
        if self.llm_client is not None and self.user_prompt_template:
            try:
                formatted_prompt = self.user_prompt_template.format(
                    paper_title=paper_title,
                    total_claims=total_claims,
                    supported_count=supported_count,
                    partially_supported_count=partially_supported_count,
                    unsupported_count=unsupported_count,
                    verdicts_breakdown=verdicts_breakdown,
                )
                if hasattr(self.llm_client, "generate"):
                    return self.llm_client.generate(
                        system_prompt=self.system_prompt,
                        prompt=formatted_prompt,
                    )
                if callable(self.llm_client):
                    return self.llm_client(formatted_prompt)
            except Exception as exc:
                logger.warning(
                    f"LLM executive summary generation failed: {exc}. Using structured fallback."
                )

        # Structured deterministic fallback (used offline, during tests, or if no LLM configured)
        return (
            f"# Executive Summary: {paper_title}\n\n"
            f"## Overall Assessment\n"
            f"The multi-agent investigation evaluated {total_claims} claim(s) across "
            f"{state.system_iteration} iteration(s).\n"
            f"- **Supported Claims:** {supported_count}\n"
            f"- **Partially Supported Claims:** {partially_supported_count}\n"
            f"- **Unsupported / Refuted Claims:** {unsupported_count}\n\n"
            f"## Claim Verdicts Breakdown\n"
            f"{verdicts_breakdown}\n"
        )
