"""Attack Agent: Investigates weaknesses, contradictory findings, and limits of a claim."""

import logging
import os
import re
from typing import Any, Dict, List, Literal, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

from mascv.agents.base import BaseAgent
from mascv.models.argument import Argument
from mascv.models.evidence import EvidenceBundle
from mascv.tools.evidence_tools import search_scientific_evidence, calculate

logger = logging.getLogger(__name__)

load_dotenv()


class AttackResult(BaseModel):
    """Structured result returned by the Attack Agent challenging a claim."""

    attack_points: List[str] = Field(
        default_factory=list,
        description="Specific reasons why the scientific claim may be wrong, overstated, or conditional.",
    )

    evidence_found: List[str] = Field(
        default_factory=list,
        description="Evidence from the search that supports the attack.",
    )

    vulnerabilities: List[str] = Field(
        default_factory=list,
        description="Scientific or methodological weaknesses found.",
    )

    strength: Literal["Strong", "Moderate", "Weak"] = Field(
        default="Moderate",
        description="Overall strength of the attack: Strong, Moderate, or Weak.",
    )


class AttackAgent(BaseAgent):
    """
    Agent 6: Investigates contradictory results, failed replications,
    and methodological limitations to formulate an adversarial counter-case.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        llm: Optional[Any] = None,
    ) -> None:
        """Initialize AttackAgent with configuration and Gemma 4 model."""
        super().__init__(name="AttackAgent", config=config)

        if llm is not None:
            self.llm = llm
        else:
            model_name = self.config.get("model", "gemma-4-26b-a4b-it")
            temperature = float(self.config.get("temperature", 0.2))
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                max_retries=2,
            )

        self.structured_llm = self.llm.with_structured_output(AttackResult)
        self.tools = [search_scientific_evidence, calculate]

    def execute(
        self,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute adversarial investigation on the active claim within InvestigationState.
        """
        active_claim_id = (
            state.get("active_claim_id")
            if isinstance(state, dict)
            else getattr(state, "active_claim_id", None)
        )
        if not active_claim_id:
            raise ValueError("state.active_claim_id is missing.")

        claims = state.get("claims", {}) if isinstance(state, dict) else getattr(state, "claims", {})
        if active_claim_id not in claims:
            raise ValueError(f"Claim {active_claim_id} not found in state.")

        claim_state = claims[active_claim_id]
        if isinstance(claim_state, dict):
            claim = claim_state.get("claim", {})
            support_arg = claim_state.get("support_argument")
            evidence_ids = claim_state.get("evidence_bundle_ids", [])
        else:
            claim = getattr(claim_state, "claim", None)
            support_arg = getattr(claim_state, "support_argument", None)
            evidence_ids = getattr(claim_state, "evidence_bundle_ids", [])

        claim_statement = (
            claim.get("statement", "") if isinstance(claim, dict) else getattr(claim, "statement", str(claim))
        )

        support_text = ""
        if support_arg:
            if hasattr(support_arg, "conclusion"):
                support_text = f"Conclusion: {support_arg.conclusion}\nPremises: {', '.join(support_arg.premises)}"
            elif isinstance(support_arg, dict):
                support_text = f"Conclusion: {support_arg.get('conclusion', '')}\nPremises: {', '.join(support_arg.get('premises', []))}"

        # Collect relevant evidence passages from global evidence store
        evidence_store = state.get("global_evidence_store", {}) if isinstance(state, dict) else getattr(state, "global_evidence_store", {})
        evidence_texts = []
        for eid in evidence_ids:
            item = evidence_store.get(eid)
            if item:
                cnt = item.get("content", "") if isinstance(item, dict) else getattr(item, "content", "")
                rel = item.get("relationship", "") if isinstance(item, dict) else getattr(item, "relationship", "")
                evidence_texts.append(f"[{eid}] ({rel}): {cnt}")

        # Search for external counter-evidence, critiques, and limitations
        external_evidence = ""
        try:
            search_query = f"{claim_statement} limitations weaknesses disadvantages failure modes"
            external_evidence = search_scientific_evidence.invoke(search_query)
        except Exception as exc:
            logger.warning("Adversarial search invocation failed: %s", exc)

        # Construct adversarial counter-case
        argument = self.construct_counter_case(
            claim_id=active_claim_id,
            claim_text=claim_statement,
            support_text=support_text,
            evidence_summary="\n".join(evidence_texts),
            external_evidence=external_evidence,
        )

        status_msg = f"Attack argument constructed (Strength: {argument.strength})."
        if isinstance(claim_state, dict):
            claim_state["attack_argument"] = argument
            claim_state["status_message"] = status_msg
            if isinstance(state, dict):
                state["claims"][active_claim_id] = claim_state
        else:
            claim_state.attack_argument = argument
            claim_state.status_message = status_msg

        return state

    def construct_counter_case(
        self,
        claim_id: str,
        evidence: Optional[List[Any]] = None,
        claim_text: str = "",
        support_text: str = "",
        evidence_summary: str = "",
        external_evidence: str = "",
    ) -> Argument:
        """
        Challenge the claim and synthesize an adversarial counterargument Argument.
        """
        prompt = f"""
You are Agent 6, the Adversarial Attack Agent in the MASCV scientific claim verification system.
Your responsibility is to critically challenge the target scientific proposition by probing
experimental edge cases, unstated hyperparameters, benchmark limitations, scaling failures,
and potential counter-evidence.

TARGET SCIENTIFIC CLAIM:
{claim_text}

PROPONENT SUPPORT CASE:
{support_text or "No proponent argument provided."}

AVAILABLE EXTRACTED EVIDENCE (TARGET PAPER):
{evidence_summary or "No counter-evidence bundles indexed."}

EXTERNAL CRITIQUES, BENCHMARKS & DISCUSSION:
{external_evidence or "No external critiques found."}

Rules:
1. Examine methodological assumptions, benchmark bounds, practical trade-offs, and potential overgeneralizations.
2. Formulate 2-3 specific, rigorous attack points based on the scientific context and external critiques.
3. Identify vulnerabilities (e.g. low-rank bottlenecking, complex reasoning degradation, memory overhead trade-offs).
4. Distinguish between evidence that directly contradicts vs. evidence that narrows the claim's scope.
5. Extract exact URLs from the external evidence into 'evidence_found' to ground the attack.
6. Provide a realistic attack strength (Strong, Moderate, or Weak).
"""

        try:
            attack_result = self.structured_llm.invoke(prompt)
        except Exception as exc:
            logger.warning("AttackAgent structured generation failed: %s. Using heuristic fallback.", exc)
            attack_result = AttackResult(
                attack_points=[
                    "Potential representation collapse when adaptation rank is constrained below task complexity bounds.",
                    "Evaluation restricted to specific baseline architectures and standardized benchmarks without domain shift stress-testing."
                ],
                evidence_found=[],
                vulnerabilities=["Performance degradation on out-of-distribution reasoning tasks."],
                strength="Moderate",
            )

        # Build canonical Argument model for dialectic synthesis
        premises = attack_result.attack_points or ["Methodological limitations exist under constrained evaluation settings."]
        vulnerabilities = attack_result.vulnerabilities or ["Boundary condition constraints."]
        conclusion = (
            f"The claim is vulnerable to boundary limitations: {'; '.join(vulnerabilities[:2])}"
        )

        # Extract cited URLs from external evidence and attack result
        cited_urls: List[str] = []
        if attack_result.evidence_found:
            for item in attack_result.evidence_found:
                urls = re.findall(r"https?://[^\s)\]]+", item)
                for u in urls:
                    if u not in cited_urls:
                        cited_urls.append(u)
        if external_evidence:
            urls = re.findall(r"https?://[^\s)\]]+", external_evidence)
            for u in urls:
                if u not in cited_urls:
                    cited_urls.append(u)

        return Argument(
            agent_name="AttackAgent",
            claim_id=claim_id,
            stance="AGAINST",
            premises=premises,
            cited_evidence_ids=cited_urls[:5],
            conclusion=conclusion,
            strength=attack_result.strength,
            identified_limitations=vulnerabilities,
        )


def run_attack_agent(paper_text: str, support: Any) -> Optional[AttackResult]:
    """
    Standalone runner for backward compatibility with standalone research scripts.
    """
    print("\n" + "=" * 60)
    print("AGENT 6 — ATTACK AGENT")
    print("=" * 60)

    agent = AttackAgent()
    claim_text = getattr(support, "claim", str(support))
    evidence_text = str(getattr(support, "supporting_evidence", ""))

    prompt = f"""
Original research paper text excerpt:
{paper_text[:3000]}

Main scientific claim:
{claim_text}

Supporting evidence identified from the paper:
{evidence_text}

Challenge this claim according to your adversarial instructions and return AttackResult.
"""

    try:
        attack = agent.structured_llm.invoke(prompt)
    except Exception as e:
        print(f"\nAgent 6 failed: {e}")
        return None

    print("\nATTACK POINTS:")
    for point in attack.attack_points:
        print(f"- {point}")

    print("\nEVIDENCE FOUND:")
    for ev in attack.evidence_found:
        print(f"- {ev}")

    print("\nVULNERABILITIES:")
    for vuln in attack.vulnerabilities:
        print(f"- {vuln}")

    print(f"\nATTACK STRENGTH: {attack.strength}")

    return attack