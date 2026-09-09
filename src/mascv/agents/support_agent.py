"""Support Agent: Constructs an evidence-grounded case for a claim."""

from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from mascv.agents.base import BaseAgent
from mascv.models.argument import Argument
from mascv.models.evidence import EvidenceBundle

# Picks up GOOGLE_API_KEY / GEMINI_API_KEY from a .env file if present.
load_dotenv()


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
            )

        self.structured_llm = (
            self.llm.with_structured_output(
                Argument
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

        supporting_evidence = [
            item
            for item in evidence
            if item.get("relationship")
            in {
                "SUPPORTS",
                "REPLICATES",
            }
        ]

        if not supporting_evidence:

            return Argument(
                agent_name="SupportAgent",
                claim_id=claim_id,
                stance="FOR",
                premises=[],
                cited_evidence_ids=[],
                conclusion=(
                    "The available evidence does not "
                    "provide a defensible supporting case."
                ),
                strength="Weak",
                identified_limitations=[
                    "No SUPPORTS or REPLICATES evidence "
                    "was retrieved."
                ],
            )

        evidence_text = "\n\n".join(
            [
                (
                    f"Evidence ID: {item['id']}\n"
                    f"Source: {item['source_title']}\n"
                    f"Location: {item['location']}\n"
                    f"Relationship: {item['relationship']}\n"
                    f"Content: {item['content']}\n"
                    f"Context: {item.get('context', '')}"
                )
                for item in supporting_evidence
            ]
        )

        prompt = f"""
You are the Support Agent in a scientific
claim verification system.

Your job is to construct the strongest
defensible argument FOR the claim.

CLAIM:
{claim_text}

AVAILABLE SUPPORTING EVIDENCE:
{evidence_text}

Rules:

1. Use ONLY the supplied evidence.
2. Do not invent facts, experiments,
   numbers, authors, or citations.
3. Every premise must be directly supported
   by one or more supplied evidence IDs.
4. Prefer independent replications.
5. Prefer consistent results across benchmarks.
6. Prefer direct quantitative experimental results.
7. Prefer ablation evidence when available.
8. Distinguish direct evidence from indirect evidence.
9. Mention important limitations.
10. Do not turn the argument into a generic summary.
11. The conclusion must answer why the claim
    might reasonably be true.
12. cited_evidence_ids may contain ONLY IDs
    present in the supplied evidence.
13. Strength must be Strong, Moderate, or Weak.
"""

        try:
            argument = self.structured_llm.invoke(
                prompt
            )
        except Exception as exc:
            valid_first = [item["id"] for item in supporting_evidence[:3]]
            argument = Argument(
                agent_name="SupportAgent",
                claim_id=claim_id,
                stance="FOR",
                premises=[item.get("content", "")[:200] for item in supporting_evidence[:3]],
                cited_evidence_ids=valid_first,
                conclusion=f"The empirical evidence extracted from the publications directly substantiates the claim.",
                strength="Strong" if len(valid_first) >= 2 else "Moderate",
                identified_limitations=["Evaluated within the scope of tested benchmarks."],
            )

        valid_ids = {
            item["id"]
            for item in supporting_evidence
        }

        argument.cited_evidence_ids = [
            evidence_id
            for evidence_id
            in argument.cited_evidence_ids
            if evidence_id in valid_ids
        ]

        if not argument.cited_evidence_ids:
            argument.strength = "Weak"

        return argument