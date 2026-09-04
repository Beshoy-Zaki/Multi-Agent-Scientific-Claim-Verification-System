"""Unit tests for ClaimAnalystAgent extraction logic and state execution."""

import json
from unittest.mock import MagicMock
import pytest
from mascv.agents.claim_analyst import ClaimAnalystAgent
from mascv.core.state import InvestigationState
from mascv.models.claim import ClaimType, ClaimStatus
from mascv.models.paper import ResearchPaper, PaperMetadata


@pytest.fixture
def agent_config():
    return {
        "agent": {
            "name": "ClaimAnalystAgent",
            "model": "gemma-4-31b-it",
            "parameters": {
                "max_claims_to_extract": 5,
                "claim_types": ["performance", "efficiency", "novelty"],
                "require_quantitative_metrics": False,
            },
        },
        "prompts": {
            "system_prompt": "Extract claims.",
            "user_prompt": "Paper: {paper_text}. Max: {max_claims}. Types: {allowed_types}",
        },
    }


@pytest.fixture
def sample_paper():
    return ResearchPaper(
        id="paper_1",
        metadata=PaperMetadata(title="Sample AI Paper"),
        raw_text="We propose SuperModel which achieves 95% accuracy on ImageNet and reduces FLOPs by 40%.",
    )


def test_claim_analyst_initialization(agent_config):
    """Test ClaimAnalystAgent initialization with configuration values."""
    agent = ClaimAnalystAgent(config=agent_config)
    assert agent.name == "ClaimAnalystAgent"
    assert agent.max_claims == 5
    assert agent.model_name == "gemma-4-31b-it"
    assert agent.require_quantitative_metrics is False


def test_extract_claims_empty_text(agent_config):
    """Empty paper text returns empty claim list without calling LLM."""
    mock_llm = MagicMock()
    agent = ClaimAnalystAgent(config=agent_config, llm_client=mock_llm)
    claims = agent.extract_claims("")
    assert claims == []
    mock_llm.generate.assert_not_called()


def test_extract_claims_no_llm_client(agent_config):
    """When no LLM client is supplied, returns empty list safely."""
    agent = ClaimAnalystAgent(config=agent_config, llm_client=None)
    claims = agent.extract_claims("Some text")
    assert claims == []


def test_extract_claims_with_mock_llm(agent_config):
    """Test full extraction pipeline with mock LLM JSON response."""
    llm_response = json.dumps([
        {
            "id": "C1",
            "subject": "SuperModel",
            "statement": "Achieves 95% top-1 accuracy on ImageNet.",
            "claim_type": "performance",
            "benchmarks": ["ImageNet-1K"],
            "metrics": ["Accuracy: 95%"],
            "comparisons": ["ResNet-50"],
        },
        {
            "id": "C2",
            "subject": "SuperModel",
            "statement": "Reduces inference FLOPs by 40%.",
            "claim_type": "efficiency",
            "benchmarks": ["ImageNet-1K"],
            "metrics": ["FLOPs: -40%"],
            "comparisons": ["ResNet-50"],
        }
    ])
    mock_llm = MagicMock()
    mock_llm.generate.return_value = f"```json\n{llm_response}\n```"

    agent = ClaimAnalystAgent(config=agent_config, llm_client=mock_llm)
    claims = agent.extract_claims("Paper text here", paper_id="paper_100")

    assert len(claims) == 2
    assert claims[0].id == "C1"
    assert claims[0].paper_id == "paper_100"
    assert claims[0].claim_type == ClaimType.PERFORMANCE
    assert claims[0].status == ClaimStatus.EXTRACTED
    assert claims[1].claim_type == ClaimType.EFFICIENCY


def test_execute_populates_state_claims(agent_config, sample_paper):
    """Execute method populates state.claims dictionary and sets metadata count."""
    llm_response = json.dumps([
        {
            "id": "C1",
            "subject": "SuperModel",
            "statement": "Achieves 95% accuracy.",
            "claim_type": "performance",
        }
    ])
    mock_llm = MagicMock()
    mock_llm.generate.return_value = llm_response

    agent = ClaimAnalystAgent(config=agent_config, llm_client=mock_llm)
    state = InvestigationState(paper=sample_paper)

    updated_state = agent.execute(state)

    assert "C1" in updated_state.claims
    assert updated_state.claims["C1"].claim.statement == "Achieves 95% accuracy."
    assert updated_state.metadata["claims_extracted_count"] == 1


def test_require_quantitative_metrics_filter(sample_paper):
    """When require_quantitative_metrics is True, non-quantitative claims are filtered out."""
    config = {
        "agent": {
            "parameters": {
                "max_claims_to_extract": 5,
                "require_quantitative_metrics": True,
            }
        }
    }
    llm_response = json.dumps([
        {
            "id": "C1",
            "subject": "Model A",
            "statement": "We propose a novel framework.",
            "claim_type": "novelty",
            "metrics": [],  # No numbers
        },
        {
            "id": "C2",
            "subject": "Model A",
            "statement": "Achieves 95% accuracy.",
            "claim_type": "performance",
            "metrics": ["95% accuracy"],  # Has numbers
        }
    ])
    mock_llm = MagicMock()
    mock_llm.generate.return_value = llm_response

    agent = ClaimAnalystAgent(config=config, llm_client=mock_llm)
    claims = agent.extract_claims(sample_paper.raw_text)

    assert len(claims) == 1
    assert claims[0].id == "C2"
