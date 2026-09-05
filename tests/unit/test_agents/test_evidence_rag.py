"""Unit tests for EvidenceRAGAgent (Agent 4)."""

from unittest.mock import MagicMock, patch

from mascv.agents.evidence_rag import EvidenceRAGAgent
from mascv.core.state import ClaimInvestigationState, InvestigationState
from mascv.models.claim import Claim, ClaimType
from mascv.models.paper import DocumentSection, PaperMetadata, ResearchPaper


def _fake_extractor(relevant=True, relationship="SUPPORTS", confidence=0.9):
    extractor = MagicMock()
    extractor.extract.return_value = MagicMock(
        relevant=relevant,
        relationship=relationship,
        evidence_text="Method X achieves 84% vs 79% for Method Y.",
        context="CIFAR-100, 200 epochs",
        confidence_score=confidence,
    )
    return extractor


def _sample_state():
    claim = Claim(
        id="C1",
        paper_id="P1",
        subject="Method X",
        statement="Method X improves accuracy.",
        claim_type=ClaimType.PERFORMANCE,
    )

    paper = ResearchPaper(
        id="P1",
        metadata=PaperMetadata(title="Test Paper"),
        raw_text="",
        sections=[
            DocumentSection(
                title="Results",
                content=(
                    "Method X achieves 84 percent accuracy on benchmark A "
                    "outperforming Method Y at 79 percent. "
                )
                * 40,
                page_number=5,
            )
        ],
    )

    state = InvestigationState(
        paper=paper,
        claims={"C1": ClaimInvestigationState(claim=claim)},
        active_claim_id="C1",
    )

    return state.model_dump()


def test_evidence_rag_initialization():
    with patch("mascv.agents.evidence_rag.EvidenceExtractor"):
        agent = EvidenceRAGAgent()

    assert agent.name == "EvidenceRAGAgent"


def test_evidence_rag_initialization_accepts_injected_dependencies():
    """The agent should be constructible without a live LLM/API key."""

    fake_extractor = _fake_extractor()
    fake_retriever = MagicMock()

    agent = EvidenceRAGAgent(retriever=fake_retriever, extractor=fake_extractor)

    assert agent.retriever is fake_retriever
    assert agent.extractor is fake_extractor


def test_evidence_rag_execute_builds_evidence_and_persists_state():
    """Regression test: evidence_bundle_ids must survive round-tripping
    through the canonical pydantic InvestigationState (state.model_dump()),
    not just plain dict state."""

    agent = EvidenceRAGAgent(extractor=_fake_extractor())

    state = _sample_state()
    out_state = agent.execute(state)

    claim_state = out_state["claims"]["C1"]

    assert len(claim_state["evidence_bundle_ids"]) > 0
    assert claim_state["status_message"].startswith("Extracted")

    # Every referenced evidence id must actually exist in the global store.
    for evidence_id in claim_state["evidence_bundle_ids"]:
        assert evidence_id in out_state["global_evidence_store"]

    stored = out_state["global_evidence_store"][claim_state["evidence_bundle_ids"][0]]
    assert stored["claim_id"] == "C1"
    assert stored["relationship"] == "SUPPORTS"


def test_evidence_rag_execute_skips_irrelevant_chunks():
    """Chunks the extractor marks irrelevant should not become evidence."""

    agent = EvidenceRAGAgent(extractor=_fake_extractor(relevant=False))

    state = _sample_state()
    out_state = agent.execute(state)

    claim_state = out_state["claims"]["C1"]
    assert claim_state["evidence_bundle_ids"] == []


def test_evidence_rag_missing_active_claim_raises():
    agent = EvidenceRAGAgent(extractor=_fake_extractor())

    state = _sample_state()
    state["active_claim_id"] = None

    try:
        agent.execute(state)
        assert False, "expected ValueError"
    except ValueError:
        pass
