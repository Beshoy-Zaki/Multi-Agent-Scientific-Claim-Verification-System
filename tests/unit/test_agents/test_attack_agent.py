"""Unit tests for AttackAgent (Agent 6)."""

from unittest.mock import MagicMock
import pytest
from mascv.agents.attack_agent import AttackAgent, AttackResult, run_attack_agent
from mascv.core.state import InvestigationState, ClaimInvestigationState
from mascv.models.claim import Claim, ClaimType
from mascv.models.argument import Argument


def _sample_state():
    claim = Claim(
        id="C1",
        paper_id="P1",
        subject="LoRA Efficiency",
        statement="LoRA reduces memory by 3x without accuracy loss.",
        claim_type=ClaimType.EFFICIENCY,
    )
    support_arg = Argument(
        agent_name="SupportAgent",
        claim_id="C1",
        stance="FOR",
        premises=["Empirical results show 3x VRAM reduction on GPT-3 175B."],
        conclusion="LoRA enables fine-tuning on consumer GPUs.",
        strength="Strong",
    )
    claim_state = ClaimInvestigationState(
        claim=claim,
        support_argument=support_arg,
    )
    state = InvestigationState(
        claims={"C1": claim_state},
        active_claim_id="C1",
    )
    return state.model_dump()


def test_attack_agent_initialization():
    fake_llm = MagicMock()
    agent = AttackAgent(llm=fake_llm)
    assert agent.name == "AttackAgent"
    assert len(agent.tools) == 2


def test_attack_agent_execute_builds_counterargument():
    mock_llm = MagicMock()
    mock_attack_result = AttackResult(
        attack_points=["Rank collapse under r < 8 on complex reasoning benchmarks."],
        evidence_found=["https://arxiv.org/abs/2406.03136"],
        vulnerabilities=["Degradation on GSM8K reasoning."],
        strength="Moderate",
    )
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = mock_attack_result
    mock_llm.with_structured_output.return_value = mock_structured

    agent = AttackAgent(llm=mock_llm)
    state = _sample_state()

    out_state = agent.execute(state)
    active_claim = out_state["claims"]["C1"]

    assert "attack_argument" in active_claim
    arg = active_claim["attack_argument"]
    assert arg.stance == "AGAINST"
    assert arg.strength == "Moderate"
    assert "Rank collapse" in arg.premises[0]
    assert "GSM8K" in arg.identified_limitations[0]


def test_attack_agent_missing_active_claim_raises():
    fake_llm = MagicMock()
    agent = AttackAgent(llm=fake_llm)
    state = _sample_state()
    state["active_claim_id"] = None

    with pytest.raises(ValueError, match="active_claim_id is missing"):
        agent.execute(state)


def test_attack_agent_independence_requires_claim_scoped_external_evidence():
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = AttackResult(
        attack_points=["[DIRECTLY EVIDENCED] A counter-study disagrees."],
        evidence_found=["https://arxiv.org/abs/2406.03136"],
        vulnerabilities=["Limited generalization."],
        strength="Moderate",
    )
    mock_llm.with_structured_output.return_value = mock_structured
    agent = AttackAgent(llm=mock_llm)

    state = _sample_state()
    state["claims"]["C1"]["evidence_bundle_ids"] = ["E-C2"]
    state["global_evidence_store"] = {
        "E-C2": {
            "id": "E-C2", "claim_id": "C2", "source_paper_id": "P2",
            "source_title": "External paper", "location": "p. 1",
            "content": "Counter-evidence for C2.", "relationship": "CONTRADICTS",
            "source_type": "EXTERNAL_SOURCE", "is_independent": True,
        }
    }

    argument = agent.execute(state)["claims"]["C1"]["attack_argument"]

    assert argument.has_independent_evidence is False
