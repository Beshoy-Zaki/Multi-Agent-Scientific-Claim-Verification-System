"""Unit tests for SupervisorAgent decision logic and state execution."""

from unittest.mock import MagicMock
import pytest
from mascv.agents.supervisor import SupervisorAgent
from mascv.core.state import InvestigationState, ClaimInvestigationState
from mascv.models.claim import Claim, ClaimType, ClaimStatus
from mascv.models.paper import ResearchPaper, PaperMetadata
from mascv.models.verdict import Verdict, VerdictType, CriticFinding


@pytest.fixture
def supervisor():
    config = {
        "agent": {
            "parameters": {
                "max_search_cycles": 3,
                "min_confidence_to_finalize": 0.70,
                "min_independent_sources": 2,
                "require_replication_check": True,
            }
        },
        "prompts": {
            "system_prompt": "You are an expert scientific peer reviewer.",
            "user_prompt": "Summary of {paper_title}: {total_claims} claims. Breakdown: {verdicts_breakdown}",
        },
    }
    return SupervisorAgent(config=config)


@pytest.fixture
def sample_paper():
    return ResearchPaper(
        id="paper_1",
        metadata=PaperMetadata(title="Sample Optimization Paper", authors=["Alice"]),
        raw_text="This paper introduces a novel optimization algorithm achieving 95% accuracy.",
    )


@pytest.fixture
def sample_claim():
    return Claim(
        id="C1",
        paper_id="paper_1",
        subject="Novel Algorithm",
        statement="Achieves 95% test accuracy on ImageNet.",
        claim_type=ClaimType.PERFORMANCE,
        status=ClaimStatus.EXTRACTED,
    )


def test_supervisor_initialization(supervisor):
    """Test that supervisor initializes with correct configuration knobs."""
    assert supervisor.name == "SupervisorAgent"
    assert supervisor.max_search_cycles == 3
    assert supervisor.min_confidence == 0.70
    assert supervisor.min_independent_sources == 2
    assert supervisor.require_replication_check is True


def test_rule_1_no_claims_routes_to_claim_analyst(supervisor, sample_paper):
    """Rule 1: If no claims exist and paper text is present, route to claim_analyst."""
    state = InvestigationState(paper=sample_paper)
    decision = supervisor.decide_next_step(state)
    assert decision == "claim_analyst"


def test_rule_1_no_claims_no_paper_routes_to_end(supervisor):
    """Rule 1: If no claims and no paper text exist, route to end."""
    state = InvestigationState()
    decision = supervisor.decide_next_step(state)
    assert decision == "end"


def test_rule_2_iteration_ceiling_routes_to_finalize(supervisor, sample_claim):
    """Rule 2: If system iteration reaches ceiling, halt and route to finalize."""
    state = InvestigationState(
        claims={"C1": ClaimInvestigationState(claim=sample_claim)},
        system_iteration=3,
        max_iterations=3,
    )
    decision = supervisor.decide_next_step(state)
    assert decision == "finalize"


def test_rule_3_insufficient_sources_routes_to_paper_search(supervisor, sample_claim):
    """Rule 3: If a claim has fewer than min_independent_sources, route to paper_search."""
    state = InvestigationState(
        claims={"C1": ClaimInvestigationState(claim=sample_claim, external_papers_found=["paper_a"])}
    )
    decision = supervisor.decide_next_step(state)
    assert decision == "paper_search"
    assert state.active_claim_id == "C1"
    assert state.claims["C1"].iteration_count == 1


def test_rule_3_low_confidence_verdict_triggers_search_loop(supervisor, sample_claim):
    """Rule 3: If confidence is below threshold, trigger paper_search again."""
    finding = CriticFinding(
        citation_valid=True,
        reasoning_sound=True,
        overgeneralization_detected=False,
        fair_comparison=True,
        critique_notes="Needs corroboration",
    )
    low_confidence_verdict = Verdict(
        claim_id="C1",
        verdict=VerdictType.PARTIALLY_SUPPORTED,
        confidence=0.55,  # below 0.70 threshold
        critic_finding=finding,
        strongest_supporting_argument="Good benchmark",
        strongest_counterargument="Small sample size",
        synthesis_summary="Inconclusive",
    )
    state = InvestigationState(
        claims={
            "C1": ClaimInvestigationState(
                claim=sample_claim,
                external_papers_found=["paper_a", "paper_b"],
                verdict=low_confidence_verdict,
            )
        }
    )
    decision = supervisor.decide_next_step(state)
    assert decision == "paper_search"
    assert state.active_claim_id == "C1"


def test_rule_4_all_finalized_routes_to_finalize(supervisor, sample_claim):
    """Rule 4: If all claims are finalized, route to finalize."""
    state = InvestigationState(
        claims={"C1": ClaimInvestigationState(claim=sample_claim, is_finalized=True)}
    )
    decision = supervisor.decide_next_step(state)
    assert decision == "finalize"


def test_execute_records_status_message_and_decision(supervisor, sample_paper):
    """Execute method updates state iteration, sets status message, and saves routing decision."""
    state = InvestigationState(paper=sample_paper)
    updated_state = supervisor.execute(state)

    assert updated_state.system_iteration == 1
    assert updated_state.metadata["current_iteration"] == 1
    assert updated_state.metadata["last_supervisor_decision"] == "claim_analyst"
    assert "status_message" in updated_state.metadata
    assert "Iteration 1/3: Routing to 'claim_analyst'" in updated_state.metadata["status_message"]


def test_finalize_generates_executive_summary_fallback(supervisor, sample_paper, sample_claim):
    """When state reaches finalize, is_completed is set and executive summary fallback is generated."""
    finding = CriticFinding(
        citation_valid=True,
        reasoning_sound=True,
        overgeneralization_detected=False,
        fair_comparison=True,
        critique_notes="Valid study",
    )
    verdict = Verdict(
        claim_id="C1",
        verdict=VerdictType.SUPPORTED,
        confidence=0.92,
        critic_finding=finding,
        strongest_supporting_argument="Replicated on multiple datasets",
        strongest_counterargument="None",
        synthesis_summary="Strongly verified by literature.",
    )
    state = InvestigationState(
        paper=sample_paper,
        claims={
            "C1": ClaimInvestigationState(
                claim=sample_claim,
                external_papers_found=["paper_a", "paper_b"],
                verdict=verdict,
                is_finalized=True,
            )
        },
    )

    updated_state = supervisor.execute(state)

    assert updated_state.is_completed is True
    assert updated_state.metadata["last_supervisor_decision"] == "finalize"
    assert "executive_summary" in updated_state.metadata
    summary = updated_state.metadata["executive_summary"]
    assert "Executive Summary: Sample Optimization Paper" in summary
    assert "**Supported Claims:** 1" in summary
    assert "Strongly verified by literature." in summary


def test_finalize_with_mock_llm(sample_paper, sample_claim):
    """When an LLM client is configured, executive summary is generated through the LLM."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "AI Generated Executive Summary: The paper is empirically valid."

    config = {
        "prompts": {
            "system_prompt": "Peer review system prompt.",
            "user_prompt": "Analyze: {paper_title}. Claims: {total_claims}. Breakdown: {verdicts_breakdown}",
        }
    }
    agent = SupervisorAgent(config=config, llm_client=mock_llm)

    state = InvestigationState(
        paper=sample_paper,
        claims={"C1": ClaimInvestigationState(claim=sample_claim, is_finalized=True)},
    )
    updated_state = agent.execute(state)

    assert updated_state.is_completed is True
    assert updated_state.metadata["executive_summary"] == "AI Generated Executive Summary: The paper is empirically valid."
    mock_llm.generate.assert_called_once()
