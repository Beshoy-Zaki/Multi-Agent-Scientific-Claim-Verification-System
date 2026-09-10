"""Support Agent: Constructs an evidence-grounded case for a claim."""

from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from mascv.agents.base import BaseAgent
from mascv.models.argument import Argument
from mascv.models.evidence import EvidenceBundle, is_independent_external_evidence
from pydantic import BaseModel, Field

# Picks up GOOGLE_API_KEY / GEMINI_API_KEY from a .env file if present.
load_dotenv()


class SupportArgumentDraft(BaseModel):
    """LLM-authored rhetoric, excluding application-owned provenance fields."""

    premises: List[str] = Field(default_factory=list)
    cited_evidence_ids: List[str] = Field(default_factory=list)
    conclusion: str = ""
    strength: str = "Moderate"
    identified_limitations: List[str] = Field(default_factory=list)


@tool
def get_supporting_evidence(
    evidence: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return evidence that supports or replicates the claim."""

    return [
        item
        for item in evidence
        if item.get("relationship")
        in {
            "SUPPORTS",
            "REPLICATES",
        }
    ]


@tool
def get_replication_evidence(
    evidence: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return independent replication evidence."""

    return [
        item
        for item in evidence
        if item.get("relationship")
        == "REPLICATES"
    ]


class SupportAgent(BaseAgent):
    """
    Constructs the strongest defensible argument
    supporting a scientific claim.
    """

    def __init__(
        self,
        config: Dict[str, Any] = None,
        llm: Optional[Any] = None,
    ) -> None:
        """
        Args:
            config: Agent configuration (flat dict or a loaded
                ``config/agents/support_agent.yaml`` with an ``agent`` key).
            llm: Optional pre-built chat model. Lets callers (and tests)
                inject a fake/mock LLM instead of requiring a live
                GOOGLE_API_KEY / GEMINI_API_KEY just to construct the agent.
        """

        super().__init__(
            name="SupportAgent",
            config=config,
        )

        if llm is not None:
            self.llm = llm
        else:
            model_name = self.config.get(
                "model",
                "gemma-4-26b-a4b-it",
            )

            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=float(
                    self.config.get(
                        "temperature",
                        0.2,
                    )
                ),
                max_retries=2,
                timeout=120.0,
            )

        self.structured_llm = (
            self.llm.with_structured_output(
                SupportArgumentDraft
            )
        )

    def execute(
        self,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:

        active_claim_id = state.get(
            "active_claim_id"
        )

        if not active_claim_id:
            raise ValueError(
                "state.active_claim_id is missing."
            )

        claim_state = state["claims"][
            active_claim_id
        ]

        if hasattr(claim_state, "model_dump"):
            claim_state = claim_state.model_dump()

        claim = claim_state["claim"]

        evidence_store = state.get(
            "global_evidence_store",
            {},
        )

        evidence = []

        for evidence_id in claim_state.get(
            "evidence_bundle_ids",
            [],
        ):

            item = evidence_store.get(
                evidence_id
            )

            if item:
                if hasattr(
                    item,
                    "model_dump",
                ):
                    item = item.model_dump()

                if item.get("claim_id") == claim["id"]:
                    evidence.append(item)

        argument = self.construct_affirmative_case(
            claim_id=claim["id"],
            claim_text=claim["statement"],
            evidence=evidence,
        )

        claim_state["support_argument"] = (
            argument
        )

        claim_state["status_message"] = (
            "Support argument constructed."
        )

        state["claims"][
            active_claim_id
        ] = claim_state

        return state

    def construct_affirmative_case(
        self,
        claim_id: str,
        claim_text: str,
        evidence: List[Dict[str, Any]],
    ) -> Argument:

        independent_supporting = [
            item
            for item in evidence
            if item.get("relationship") in {"SUPPORTS", "REPLICATES"}
            and is_independent_external_evidence(item, claim_id)
        ]

        internal_supporting = [
            item
            for item in evidence
            if item.get("relationship") in {"SUPPORTS", "REPLICATES"}
            and not is_independent_external_evidence(item, claim_id)
        ]

        # Critical: When zero independent external evidence was discovered
        if not independent_supporting:
            if internal_supporting:
                target_ids = [item["id"] for item in internal_supporting[:3]]
                return Argument(
                    agent_name="SupportAgent",
                    claim_id=claim_id,
                    stance="FOR",
                    premises=[
                        f"[TARGET PAPER INTERNAL] {item.get('content', '')[:250]}"
                        for item in internal_supporting[:2]
                    ],
                    cited_evidence_ids=target_ids,
                    conclusion=(
                        "The target paper provides internal empirical assertions for this proposition, "
                        "but NO independent external literature or replication evidence was retrieved. "
                        "Independent empirical support cannot be established from the target paper alone."
                    ),
                    strength="Weak",
                    identified_limitations=[
                        "Lacks independent external verification; proposition relies solely on internal target-paper assertions."
                    ],
                    has_independent_evidence=False,
                    evidence_types_used=["TARGET_PAPER_INTERNAL"],
                )
            else:
                return Argument(
                    agent_name="SupportAgent",
                    claim_id=claim_id,
                    stance="FOR",
                    premises=[],
                    cited_evidence_ids=[],
                    conclusion=(
                        "The available evidence does not provide a defensible supporting case. "
                        "No independent external literature or replication was found."
                    ),
                    strength="Weak",
                    identified_limitations=[
                        "No SUPPORTS or REPLICATES evidence was retrieved from independent literature."
                    ],
                    has_independent_evidence=False,
                    evidence_types_used=[],
                )

        all_supporting = independent_supporting + internal_supporting
        evidence_entries = []
        for item in all_supporting:
            prov = "INDEPENDENT EXTERNAL SOURCE" if is_independent_external_evidence(item, claim_id) else "TARGET PAPER INTERNAL"
            evidence_entries.append(
                f"Evidence ID: {item['id']}\n"
                f"Provenance: [{prov}]\n"
                f"Source: {item['source_title']}\n"
                f"Location: {item['location']}\n"
                f"Relationship: {item['relationship']}\n"
                f"Content: {item['content']}\n"
                f"Context: {item.get('context', '')}"
            )
        evidence_text = "\n\n".join(evidence_entries)

        prompt = f"""
You are the Support Agent in a scientific claim verification system.
Your responsibility is to construct the strongest defensible affirmative case FOR the claim.

CRITICAL MANDATE:
1. Distinguish between [INDEPENDENT EXTERNAL SOURCE] and [TARGET PAPER INTERNAL].
   - Target paper internal passages represent the authors' own assertions.
   - Independent external sources represent external replication or peer corroboration.
   - You MUST NOT cite target paper internal passages as independent confirmation.
2. Label every premise with its provenance: [INDEPENDENT REPLICATION], [TARGET PAPER INTERNAL], or [INFERENCE].
3. Use ONLY the supplied evidence. Do not invent facts, numbers, or citations.
4. cited_evidence_ids may contain ONLY IDs present in the supplied evidence.
5. Provide a realistic strength: Strong, Moderate, or Weak.

CLAIM UNDER INVESTIGATION:
{claim_text}

AVAILABLE SUPPORTING EVIDENCE:
{evidence_text}
"""

        try:
            draft = self.structured_llm.invoke(
                prompt
            )
            argument = Argument(
                agent_name="SupportAgent",
                claim_id=claim_id,
                stance="FOR",
                premises=draft.premises,
                cited_evidence_ids=draft.cited_evidence_ids,
                conclusion=draft.conclusion,
                strength=draft.strength,
                identified_limitations=draft.identified_limitations,
                has_independent_evidence=True,
                evidence_types_used=["EXTERNAL_EMPIRICAL"],
            )
        except Exception as exc:
            valid_first = [item["id"] for item in all_supporting[:3]]
            argument = Argument(
                agent_name="SupportAgent",
                claim_id=claim_id,
                stance="FOR",
                premises=[
                    f"[{'INDEPENDENT REPLICATION' if is_independent_external_evidence(item, claim_id) else 'TARGET PAPER INTERNAL'}] {item.get('content', '')[:200]}"
                    for item in all_supporting[:2]
                ],
                cited_evidence_ids=valid_first,
                conclusion="Independent external literature reports observations consistent with the proposition, subject to verification.",
                strength="Moderate" if len(valid_first) >= 2 else "Weak",
                identified_limitations=["Automated argument synthesis fallback invoked; limited to direct passage matching."],
                has_independent_evidence=True,
                evidence_types_used=["EXTERNAL_EMPIRICAL"],
            )

        valid_ids = {
            item["id"]
            for item in all_supporting
        }

        argument.cited_evidence_ids = [
            evidence_id
            for evidence_id
            in argument.cited_evidence_ids
            if evidence_id in valid_ids
        ]

        if not argument.cited_evidence_ids:
            argument.strength = "Weak"

        argument.has_independent_evidence = True
        argument.evidence_types_used = ["EXTERNAL_EMPIRICAL"]
        if internal_supporting:
            argument.evidence_types_used.append("TARGET_PAPER_INTERNAL")

        return argument
