"""Critic Agent: Evaluates dialectic debate, validates citations, and issues final verdicts."""

import json
import logging
import os
from typing import Any, Dict, List, Literal, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from langchain_google_genai import ChatGoogleGenerativeAI

from mascv.agents.base import BaseAgent
from mascv.models.argument import Argument
from mascv.models.verdict import Verdict, VerdictType, CriticFinding
from mascv.tools.evidence_tools import calculate
from mascv.utils.text_processing import extract_json_from_text

logger = logging.getLogger(__name__)

load_dotenv()


class CriticResult(BaseModel):
    """Structured result returned by Critic Agent evaluating the dialectic arguments."""

    citation_grounding: str = Field(
        default="Direct empirical alignment with cited methodology.",
        description="Explain whether the evidence supports the Support and Attack arguments.",
    )

    experimental_parity: str = Field(
        default="Comparable experimental configurations and baseline models.",
        description="Explain whether the compared studies had sufficiently similar conditions.",
    )

    generalization: str = Field(
        default="Valid within evaluated experimental conditions.",
        description="Explain whether the claim can be generalized beyond the studied conditions.",
    )

    comparison: str = Field(
        default="Directly comparable quantitative metrics.",
        description="Explain whether the Support and Attack evidence is genuinely comparable.",
    )

    verdict: Literal["Supported", "Partially Supported", "Unsupported", "Inconclusive"] = Field(
        default="Inconclusive",
        description="Choose: Supported, Partially Supported, Unsupported, or Inconclusive.",
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the verdict from 0.0 to 1.0.",
    )

    key_issue: str = Field(
        default="Methodological constraints and comparative evaluation parity.",
        description="State the most important issue affecting the verdict.",
    )

    winner: Literal["Support", "Attack", "Neither"] = Field(
        default="Neither",
        description="Choose: Support, Attack, or Neither.",
    )

    overall_summary: str = Field(
        default="The affirmative case is evaluated against empirical counterpoints.",
        description="Compare the Support and Attack arguments and explain which is stronger.",
    )

    final_assessment: str = Field(
        default="The scientific claim is assessed against available empirical evidence.",
        description="Give the final scientific assessment and explain the verdict.",
    )

    sources: List[str] = Field(
        default_factory=list,
        description="Exact URLs of scientific sources used by the Attack Agent.",
    )

    @field_validator("verdict", mode="before")
    @classmethod
    def normalize_verdict(cls, v: Any) -> str:
        if isinstance(v, str):
            s = v.strip().title()
            if "Partially" in s:
                return "Partially Supported"
            if "Unsupported" in s or "Refuted" in s:
                return "Unsupported"
            if "Inconclusive" in s:
                return "Inconclusive"
            if "Support" in s:
                return "Supported"
        return "Inconclusive"

    @field_validator("winner", mode="before")
    @classmethod
    def normalize_winner(cls, v: Any) -> str:
        if isinstance(v, str):
            s = v.strip().title()
            if "Attack" in s:
                return "Attack"
            if "Support" in s:
                return "Support"
            if "Neither" in s:
                return "Neither"
        return "Neither"

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, v: Any) -> float:
        try:
            val = float(v)
            return max(0.0, min(1.0, val))
        except Exception:
            return 0.0


class CriticAgent(BaseAgent):
    """
    Agent 7: Evaluates opposing arguments, validates citation grounding,
    and synthesizes the final scientific verdict.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        llm: Optional[Any] = None,
    ) -> None:
        """Initialize CriticAgent with configuration and Gemma 4 model."""
        super().__init__(name="CriticAgent", config=config)

        if llm is not None:
            self.llm = llm
            self.llm_client = None
        else:
            model_name = self.config.get("model", "gemma-4-26b-a4b-it")
            temperature = float(self.config.get("temperature", 0.1))
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                max_retries=2,
                timeout=120.0,
            )
            from mascv.utils.llm import LLMClient
            self.llm_client = LLMClient(
                model_name=model_name,
                temperature=temperature,
                thinking_level="minimal",
            )

        if hasattr(self.llm, "with_structured_output"):
            try:
                self.structured_llm = self.llm.with_structured_output(CriticResult)
            except Exception:
                self.structured_llm = None
        else:
            self.structured_llm = None

        self.tools = [calculate]

    def execute(
        self,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate the dialectic debate on the active claim and record a finalized Verdict in state.
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
            attack_arg = claim_state.get("attack_argument")
        else:
            claim = getattr(claim_state, "claim", None)
            support_arg = getattr(claim_state, "support_argument", None)
            attack_arg = getattr(claim_state, "attack_argument", None)

        claim_statement = (
            claim.get("statement", "") if isinstance(claim, dict) else getattr(claim, "statement", str(claim))
        )

        # Inspect whether support argument has independent empirical evidence
        support_has_independent = False
        if support_arg:
            if hasattr(support_arg, "has_independent_evidence"):
                support_has_independent = support_arg.has_independent_evidence is True
            elif isinstance(support_arg, dict):
                support_has_independent = support_arg.get("has_independent_evidence") is True

        # Format support text
        support_summary = "No affirmative argument available."
        if support_arg:
            if hasattr(support_arg, "conclusion"):
                support_summary = f"Stance: {support_arg.stance}, Strength: {support_arg.strength}\nConclusion: {support_arg.conclusion}\nPremises: {', '.join(support_arg.premises)}"
            elif isinstance(support_arg, dict):
                support_summary = f"Stance: {support_arg.get('stance', '')}, Strength: {support_arg.get('strength', '')}\nConclusion: {support_arg.get('conclusion', '')}\nPremises: {', '.join(support_arg.get('premises', []))}"

        # Format attack text
        attack_summary = "No adversarial counterargument available."
        if attack_arg:
            if hasattr(attack_arg, "conclusion"):
                attack_summary = f"Stance: {attack_arg.stance}, Strength: {attack_arg.strength}\nConclusion: {attack_arg.conclusion}\nPremises: {', '.join(attack_arg.premises)}"
            elif isinstance(attack_arg, dict):
                attack_summary = f"Stance: {attack_arg.get('stance', '')}, Strength: {attack_arg.get('strength', '')}\nConclusion: {attack_arg.get('conclusion', '')}\nPremises: {', '.join(attack_arg.get('premises', []))}"

        # Synthesize final verdict
        verdict = self.synthesize_verdict(
            claim_id=active_claim_id,
            claim_text=claim_statement,
            support_summary=support_summary,
            attack_summary=attack_summary,
            support_arg=support_arg if isinstance(support_arg, Argument) else None,
            attack_arg=attack_arg if isinstance(attack_arg, Argument) else None,
            support_has_independent=support_has_independent,
        )

        status_msg = f"Verdict formulated: {verdict.verdict.value} (Confidence: {verdict.confidence:.2f})."
        if isinstance(claim_state, dict):
            claim_state["verdict"] = verdict
            claim_state["is_finalized"] = True
            claim_state["status_message"] = status_msg
            if isinstance(state, dict):
                state["claims"][active_claim_id] = claim_state
        else:
            claim_state.verdict = verdict
            claim_state.is_finalized = True
            claim_state.status_message = status_msg

        return state

    def synthesize_verdict(
        self,
        claim_id: str,
        claim_text: str = "",
        support_summary: str = "",
        attack_summary: str = "",
        support_arg: Optional[Argument] = None,
        attack_arg: Optional[Argument] = None,
        support_has_independent: bool = True,
    ) -> Verdict:
        """
        Weigh the proponent case vs adversarial case and synthesize the scientific verdict.
        """
        prompt = f"""
You are Agent 7, the Scientific Critic and Verdict Judge in the MASCV verification system.
Your responsibility is to impartially adjudicate the dialectic debate between the Support Agent
and Attack Agent regarding the following scientific claim:

CLAIM UNDER INVESTIGATION:
{claim_text or claim_id}

SUPPORT ARGUMENT:
{support_summary or (str(support_arg) if support_arg else "None")}

ATTACK ARGUMENT:
{attack_summary or (str(attack_arg) if attack_arg else "None")}

INDEPENDENT EVIDENCE STATUS:
Support has independent external evidence: {support_has_independent}

CRITICAL SCIENTIFIC EVALUATION RULES:
1. NON-CIRCULARITY (Target Paper Cannot Validate Itself):
   A claim cannot be validated solely by citations or data from the target paper under evaluation.
   If support has independent external evidence is False, independent replication is absent.
   Under NO CIRCUMSTANCES may you assign "Supported" without independent external corroboration.
   The maximum verdict for a claim lacking independent external evidence is "Inconclusive"
   (or "Partially Supported" only if strictly scoped to target-paper internal consistency).

2. EVIDENCE OVER RHETORIC:
   Evaluate claims based on empirical evidence rather than rhetorical persuasiveness.
   Downweight purely theoretical or speculative objections that cite no empirical data.
   Weight [INDEPENDENT REPLICATION] and [DIRECTLY EVIDENCED] premises highest.
   Treat [TARGET PAPER INTERNAL] premises as unverified hypotheses unless externally confirmed.

3. ADJUDICATION CRITERIA:
   - Citation Grounding: Does the cited evidence legitimately support the claims?
   - Experimental Parity: Are baseline models, datasets, and conditions genuinely comparable?
   - Generalization: Does the empirical evidence justify the claimed scope, or is it overgeneralized?
   - Comparison: Are the arguments directly engaging with the same proposition?

Select exactly one Verdict:
- "Supported": Claim is firmly backed by independent empirical findings from external literature with high confidence.
- "Partially Supported": Claim holds true under specific constraints, subsets of metrics, or internal validation only.
- "Unsupported": Empirical evidence refutes the claim or reveals critical failures.
- "Inconclusive": Insufficient, missing, or contradictory independent external data prevents a decisive verdict.
"""

        json_prompt = (
            prompt
            + "\n\nFormat your response as a valid JSON object matching this schema exactly:\n"
            "{\n"
            '  "citation_grounding": "string",\n'
            '  "experimental_parity": "string",\n'
            '  "generalization": "string",\n'
            '  "comparison": "string",\n'
            '  "verdict": "Supported" | "Partially Supported" | "Unsupported" | "Inconclusive",\n'
            '  "confidence": float (0.0 to 1.0),\n'
            '  "key_issue": "string",\n'
            '  "winner": "Support" | "Attack" | "Neither",\n'
            '  "overall_summary": "string",\n'
            '  "final_assessment": "string",\n'
            '  "sources": ["url1", "url2"]\n'
            "}\n"
        )

        critic_result = None

        # 1. Try structured LLM if available (used in unit tests and structured chains)
        if self.structured_llm is not None:
            try:
                res = self.structured_llm.invoke(prompt)
                if isinstance(res, CriticResult):
                    critic_result = res
                elif isinstance(res, dict):
                    critic_result = CriticResult(**res)
            except Exception as exc:
                logger.debug("CriticAgent structured_llm failed: %s", exc)

        # 2. Try LLMClient if available (production path)
        if critic_result is None and self.llm_client is not None:
            try:
                raw_text = self.llm_client.generate(prompt=json_prompt)
                json_str = extract_json_from_text(raw_text)
                data = json.loads(json_str) if json_str else {}
                critic_result = CriticResult(**data)
            except Exception as exc:
                logger.info("CriticAgent LLMClient generation failed: %s. Trying fallback LLM.", exc)

        # 3. Try raw self.llm
        if critic_result is None and self.llm is not None:
            try:
                raw_response = self.llm.invoke(json_prompt)
                content = getattr(raw_response, "content", raw_response)
                if isinstance(content, list):
                    text_parts = [
                        p.get("text", "")
                        for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    ]
                    raw_text = "\n".join(text_parts) if text_parts else str(content)
                else:
                    raw_text = str(content)

                json_str = extract_json_from_text(raw_text)
                data = json.loads(json_str) if json_str else {}
                critic_result = CriticResult(**data)
            except Exception as raw_exc:
                logger.warning("CriticAgent raw LLM failed: %s. Using heuristic fallback.", raw_exc)

        # 4. Fallback if still None
        if critic_result is None:
            critic_result = CriticResult(
                citation_grounding="Automated critique could not verify citation grounding due to model inference failure.",
                experimental_parity="Comparative experimental conditions could not be reliably determined.",
                generalization="Generalization bounds could not be evaluated automatically.",
                comparison="Direct comparison between arguments was inconclusive.",
                verdict="Inconclusive",
                confidence=0.0,
                key_issue="Automated critique fallback invoked due to model extraction failure.",
                winner="Neither",
                overall_summary="The dialectic debate could not be reliably adjudicated due to model failure.",
                final_assessment="Inconclusive: Automated synthesis failed to parse model outputs. Requires manual scientific inspection.",
                sources=[],
            )

        # Enforce Rule 1 programmatically: No independent evidence -> cannot be Supported
        if not support_has_independent and critic_result.verdict == "Supported":
            critic_result.verdict = "Inconclusive"
            critic_result.confidence = min(critic_result.confidence, 0.5)
            critic_result.final_assessment = (
                f"Inconclusive (Absence of Independent Replication): The claim relies exclusively on target paper "
                f"internal evidence. Independent external verification could not be established. "
                f"{critic_result.final_assessment}"
            )

        # Map string verdict to VerdictType enum
        v_map = {
            "Supported": VerdictType.SUPPORTED,
            "Partially Supported": VerdictType.PARTIALLY_SUPPORTED,
            "Unsupported": VerdictType.UNSUPPORTED,
            "Inconclusive": VerdictType.INCONCLUSIVE,
        }
        verdict_type = v_map.get(critic_result.verdict, VerdictType.INCONCLUSIVE)

        is_fallback = critic_result.verdict == "Inconclusive" and critic_result.confidence == 0.0
        finding = CriticFinding(
            citation_valid=not is_fallback,
            reasoning_sound=not is_fallback,
            overgeneralization_detected=(
                "limit" in critic_result.generalization.lower()
                or "overgeneral" in critic_result.generalization.lower()
            ),
            fair_comparison=not is_fallback,
            critique_notes=critic_result.key_issue,
        )

        return Verdict(
            claim_id=claim_id,
            verdict=verdict_type,
            confidence=critic_result.confidence,
            critic_finding=finding,
            strongest_supporting_argument=support_summary[:300],
            strongest_counterargument=attack_summary[:300],
            synthesis_summary=critic_result.final_assessment,
        )

    def validate_citations(
        self,
        argument: Argument,
        evidence_store: Dict[str, Any],
    ) -> bool:
        """
        Verify that cited evidence identifiers legitimately exist in the global evidence store.
        """
        if not argument or not argument.cited_evidence_ids:
            return True
        return all(eid in evidence_store for eid in argument.cited_evidence_ids)


def run_critic_agent(
    paper_text: str,
    support: Any,
    attack: Any,
) -> Optional[CriticResult]:
    """
    Standalone runner for backward compatibility with standalone research scripts.
    """
    print("\n" + "=" * 60)
    print("AGENT 7 — CRITIC AGENT")
    print("=" * 60)

    agent = CriticAgent()
    claim_text = getattr(support, "claim", str(support))
    support_evidence = getattr(support, "supporting_evidence", "")
    attack_points = getattr(attack, "attack_points", [])
    evidence_found = getattr(attack, "evidence_found", [])
    vulnerabilities = getattr(attack, "vulnerabilities", [])
    attack_strength = getattr(attack, "strength", "Moderate")

    prompt = f"""
Original research paper excerpt:
{paper_text[:2000]}

Original scientific claim:
{claim_text}

Supporting evidence:
{support_evidence}

Attack points:
{attack_points}

External evidence from Attack Agent:
{evidence_found}

Vulnerabilities:
{vulnerabilities}

Attack strength:
{attack_strength}

Evaluate these inputs according to your instructions and return the complete CriticResult.
"""

    try:
        critic = agent.structured_llm.invoke(prompt)
    except Exception as e:
        print(f"\nAgent 7 failed: {e}")
        return None

    print(f"\nVERDICT: {critic.verdict}")
    print(f"CONFIDENCE: {critic.confidence}")
    print(f"KEY ISSUE: {critic.key_issue}")
    print(f"WINNER: {critic.winner}")
    print(f"FINAL ASSESSMENT: {critic.final_assessment}")

    return critic
