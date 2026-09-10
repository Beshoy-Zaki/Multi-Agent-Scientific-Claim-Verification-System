"""Unit tests for SupportAgent (Agent 5)."""

from unittest.mock import MagicMock

from mascv.agents.support_agent import SupportAgent
from mascv.models.argument import Argument


def test_argument_schema():

    argument = Argument(
        agent_name="SupportAgent",
        claim_id="C1",
        stance="FOR",
        premises=[
            "The proposed method improved accuracy."
        ],
        cited_evidence_ids=[
            "E-001"
        ],
        conclusion=(
            "The available evidence provides "
            "support for the claim."
        ),
        strength="Moderate",
        identified_limitations=[
            "Only one benchmark was evaluated."
        ],
    )

    assert argument.agent_name == "SupportAgent"

    assert argument.claim_id == "C1"

    assert "E-001" in (
        argument.cited_evidence_ids
    )

    assert argument.strength in {
        "Strong",
        "Moderate",
        "Weak",
    }


def _fake_llm(returned_argument):
    llm = MagicMock()
    structured = MagicMock()
    structured.invoke.return_value = returned_argument
    llm.with_structured_output.return_value = structured
    return llm


def _evidence_store():
    return {
        "E-1": {
            "id": "E-1",
            "claim_id": "C1",
            "source_paper_id": "P1",
            "source_title": "Test Paper",
            "location": "Page 5",
            "content": "Method X: 84% vs Method Y: 79%.",
            "context": "CIFAR-100",
            "relationship": "SUPPORTS",
            "confidence_score": 0.9,
            "source_type": "EXTERNAL_SOURCE",
            "is_independent": True,
        },
        "E-2": {
            "id": "E-2",
            "claim_id": "C1",
            "source_paper_id": "P2",
            "source_title": "Replication Study",
            "location": "Page 2",
            "content": "Independent replication confirms the improvement.",
            "context": "ImageNet",
            "relationship": "REPLICATES",
            "confidence_score": 0.85,
            "source_type": "EXTERNAL_SOURCE",
            "is_independent": True,
        },
        "E-3": {
            "id": "E-3",
            "claim_id": "C1",
            "source_paper_id": "P3",
            "source_title": "Critique Paper",
            "location": "Page 9",
            "content": "No improvement observed under harder conditions.",
            "context": "OOD benchmark",
            "relationship": "CONTRADICTS",
            "confidence_score": 0.8,
            "source_type": "EXTERNAL_SOURCE",
            "is_independent": True,
        },
    }


def _base_state():
    return {
        "active_claim_id": "C1",
        "claims": {
            "C1": {
                "claim": {"id": "C1", "statement": "Method X improves accuracy."},
                "evidence_bundle_ids": ["E-1", "E-2", "E-3"],
            }
        },
        "global_evidence_store": _evidence_store(),
    }


def test_support_agent_initialization_accepts_injected_llm():
    """The agent should be constructible without a live GOOGLE_API_KEY."""

    fake_llm = _fake_llm(None)
    agent = SupportAgent(llm=fake_llm)

    assert agent.name == "SupportAgent"
    assert agent.llm is fake_llm


def test_support_agent_uses_only_supporting_and_replicating_evidence():
    """CONTRADICTS evidence must never be handed to the LLM as support."""

    seen_prompts = []

    def fake_invoke(prompt):
        seen_prompts.append(prompt)
        return Argument(
            agent_name="SupportAgent",
            claim_id="C1",
            stance="FOR",
            premises=["Two independent sources report the improvement."],
            cited_evidence_ids=["E-1", "E-2"],
            conclusion="The claim is plausible given consistent evidence.",
            strength="Strong",
            identified_limitations=[],
        )

    llm = MagicMock()
    structured = MagicMock()
    structured.invoke.side_effect = fake_invoke
    llm.with_structured_output.return_value = structured

    agent = SupportAgent(llm=llm)

    state = _base_state()
    out_state = agent.execute(state)

    assert "E-3" not in seen_prompts[0]
    assert "E-1" in seen_prompts[0]
    assert "E-2" in seen_prompts[0]

    argument = out_state["claims"]["C1"]["support_argument"]
    assert argument.cited_evidence_ids == ["E-1", "E-2"]
    assert out_state["claims"]["C1"]["status_message"] == "Support argument constructed."


def test_support_agent_strips_hallucinated_evidence_ids():
    """cited_evidence_ids returned by the LLM must be filtered against the
    supplied supporting evidence -- an id the LLM invents must be dropped,
    and if nothing valid remains the strength is downgraded to Weak."""

    hallucinated = Argument(
        agent_name="SupportAgent",
        claim_id="C1",
        stance="FOR",
        premises=["invented premise"],
        cited_evidence_ids=["E-999"],
        conclusion="invented conclusion",
        strength="Strong",
        identified_limitations=[],
    )

    agent = SupportAgent(llm=_fake_llm(hallucinated))

    state = _base_state()
    out_state = agent.execute(state)

    argument = out_state["claims"]["C1"]["support_argument"]
    assert argument.cited_evidence_ids == []
    assert argument.strength == "Weak"


def test_support_agent_no_supporting_evidence_short_circuits_llm():
    """When there's no SUPPORTS/REPLICATES evidence, the agent must return a
    Weak argument WITHOUT ever calling the LLM."""

    llm = MagicMock()
    agent = SupportAgent(llm=llm)

    state = _base_state()
    state["claims"]["C1"]["evidence_bundle_ids"] = ["E-3"]  # only CONTRADICTS

    out_state = agent.execute(state)
    argument = out_state["claims"]["C1"]["support_argument"]

    assert argument.strength == "Weak"
    assert argument.cited_evidence_ids == []
    llm.with_structured_output.return_value.invoke.assert_not_called()


def test_support_agent_missing_active_claim_raises():
    agent = SupportAgent(llm=_fake_llm(None))

    state = _base_state()
    state["active_claim_id"] = None

    try:
        agent.execute(state)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_support_agent_zero_independent_evidence_sets_weak_and_false_flag():
    """When only TARGET_PAPER internal evidence is present, SupportAgent flags has_independent_evidence=False."""
    llm = MagicMock()
    agent = SupportAgent(llm=llm)

    internal_store = {
        "E-TARGET": {
            "id": "E-TARGET",
            "claim_id": "C1",
            "source_paper_id": "TARGET_P1",
            "source_title": "Target Paper Itself",
            "location": "Section 4.1",
            "content": "We achieve 98% accuracy on our custom dataset.",
            "context": "Internal Evaluation",
            "relationship": "SUPPORTS",
            "confidence_score": 0.95,
            "source_type": "TARGET_PAPER",
            "is_independent": False,
        }
    }

    state = {
        "active_claim_id": "C1",
        "claims": {
            "C1": {
                "claim": {"id": "C1", "statement": "Target algorithm achieves 98% accuracy."},
                "evidence_bundle_ids": ["E-TARGET"],
            }
        },
        "global_evidence_store": internal_store,
    }

    out_state = agent.execute(state)
    argument = out_state["claims"]["C1"]["support_argument"]

    assert argument.has_independent_evidence is False
    assert argument.strength == "Weak"
    assert "target paper alone" in argument.conclusion.lower()
    assert any("[TARGET PAPER INTERNAL]" in p for p in argument.premises)


def test_support_agent_rejects_evidence_from_another_claim():
    """A valid external source for C2 must not be counted as C1 support."""
    llm = MagicMock()
    agent = SupportAgent(llm=llm)
    state = _base_state()
    state["global_evidence_store"] = {
        "E-C2": {
            "id": "E-C2",
            "claim_id": "C2",
            "source_paper_id": "P2",
            "source_title": "External replication for a different claim",
            "location": "p. 4",
            "content": "This evidence belongs only to C2.",
            "relationship": "SUPPORTS",
            "source_type": "EXTERNAL_SOURCE",
            "is_independent": True,
        }
    }
    state["claims"]["C1"]["evidence_bundle_ids"] = ["E-C2"]

    argument = agent.execute(state)["claims"]["C1"]["support_argument"]

    assert argument.has_independent_evidence is False
    assert argument.cited_evidence_ids == []
    llm.with_structured_output.return_value.invoke.assert_not_called()

