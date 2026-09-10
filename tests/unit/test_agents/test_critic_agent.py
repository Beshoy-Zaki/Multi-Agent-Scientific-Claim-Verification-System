"""Unit tests for CriticAgent (Agent 7)."""

from unittest.mock import MagicMock
import pytest
from mascv.agents.critic_agent import CriticAgent, CriticResult
from mascv.core.state import InvestigationState, ClaimInvestigationState
from mascv.models.claim import Claim, ClaimType
from mascv.models.argument import Argument
from mascv.models.verdict import VerdictType


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
        has_independent_evidence=True,
    )
    attack_arg = Argument(
        agent_name="AttackAgent",
        claim_id="C1",
        stance="AGAINST",
        premises=["Rank collapse occurs when r is too low on math reasoning."],
        conclusion="LoRA may fail on complex reasoning tasks.",
        strength="Moderate",
    )
    claim_state = ClaimInvestigationState(
        claim=claim,
        support_argument=support_arg,
        attack_argument=attack_arg,
    )
    state = InvestigationState(
        claims={"C1": claim_state},
        active_claim_id="C1",
    )
    return state.model_dump()


def test_critic_agent_initialization():
    fake_llm = MagicMock()
    agent = CriticAgent(llm=fake_llm)
    assert agent.name == "CriticAgent"
    assert len(agent.tools) == 1


def test_critic_agent_execute_synthesizes_verdict():
    mock_llm = MagicMock()
    mock_critic_result = CriticResult(
        citation_grounding="Empirical evidence cited directly from LoRA paper.",
        experimental_parity="Equivalent downstream evaluation tasks.",
        generalization="Valid across evaluated transformer architectures.",
        comparison="Direct evaluation of parameter efficiency.",
        verdict="Supported",
        confidence=0.92,
        key_issue="Memory savings are unambiguous and replicate across setups.",
        winner="Support",
        overall_summary="The parameter reduction and memory reduction are well verified.",
        final_assessment="The claim is supported by solid empirical evidence.",
        sources=["https://arxiv.org/abs/2106.09685"],
    )
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = mock_critic_result
    mock_llm.with_structured_output.return_value = mock_structured

    agent = CriticAgent(llm=mock_llm)
    state = _sample_state()

    out_state = agent.execute(state)
    active_claim = out_state["claims"]["C1"]

    assert "verdict" in active_claim
    verdict = active_claim["verdict"]
    assert verdict.verdict == VerdictType.SUPPORTED
    assert verdict.confidence == 0.92
    assert active_claim["is_finalized"] is True
    assert verdict.critic_finding is not None
    assert verdict.critic_finding.citation_valid is True
    assert verdict.critic_finding.reasoning_sound is True
    assert verdict.critic_finding.critique_notes == "Memory savings are unambiguous and replicate across setups."


def test_critic_agent_execute_partially_supported():
    mock_llm = MagicMock()
    mock_critic_result = CriticResult(
        citation_grounding="Partial evidence.",
        experimental_parity="Baselines differ.",
        generalization="Only tested on GLUE.",
        comparison="Metrics differ.",
        verdict="Partially Supported",
        confidence=0.70,
        key_issue="Generalization to other models is limited.",
        winner="Neither",
        overall_summary="Claim holds only for subset of tests.",
        final_assessment="Partially substantiated.",
    )
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = mock_critic_result
    mock_llm.with_structured_output.return_value = mock_structured

    agent = CriticAgent(llm=mock_llm)
    state = _sample_state()

    out_state = agent.execute(state)
    verdict = out_state["claims"]["C1"]["verdict"]
    assert verdict.verdict == VerdictType.PARTIALLY_SUPPORTED
    assert verdict.confidence == 0.70


def test_critic_agent_missing_active_claim_raises():
    fake_llm = MagicMock()
    agent = CriticAgent(llm=fake_llm)
    state = _sample_state()
    state["active_claim_id"] = None

    with pytest.raises(ValueError, match="active_claim_id is missing"):
        agent.execute(state)


def test_critic_agent_fallback_on_llm_exception():
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = RuntimeError("API connection timeout")
    mock_llm.with_structured_output.return_value = mock_structured

    agent = CriticAgent(llm=mock_llm)
    state = _sample_state()

    out_state = agent.execute(state)
    active_claim = out_state["claims"]["C1"]
    assert "verdict" in active_claim
    assert active_claim["is_finalized"] is True
    assert active_claim["verdict"].verdict == VerdictType.INCONCLUSIVE
    assert active_claim["verdict"].confidence == 0.0


def test_critic_agent_downgrades_to_inconclusive_when_no_independent_evidence():
    """Rule 1 & Rule 2: If support has no independent external evidence, verdict cannot be Supported."""
    mock_llm = MagicMock()
    # Mock LLM returns "Supported"
    mock_critic_result = CriticResult(
        citation_grounding="Grounding looks plausible internally.",
        experimental_parity="Consistent with paper.",
        generalization="Standard transformer benchmarks.",
        comparison="Direct.",
        verdict="Supported",
        confidence=0.95,
        key_issue="Paper self-reports significant efficiency.",
        winner="Support",
        overall_summary="The claim seems internally supported.",
        final_assessment="Supported by internal findings.",
    )
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = mock_critic_result
    mock_llm.with_structured_output.return_value = mock_structured

    agent = CriticAgent(llm=mock_llm)
    state = _sample_state()

    # Explicitly set has_independent_evidence = False on the support argument
    support_arg = state["claims"]["C1"]["support_argument"]
    support_arg["has_independent_evidence"] = False
    state["claims"]["C1"]["support_argument"] = support_arg

    out_state = agent.execute(state)
    verdict = out_state["claims"]["C1"]["verdict"]

    # Must be downgraded to INCONCLUSIVE to prevent circular target-paper validation
    assert verdict.verdict == VerdictType.INCONCLUSIVE
    assert verdict.confidence <= 0.5
    assert "Absence of Independent Replication" in verdict.synthesis_summary


