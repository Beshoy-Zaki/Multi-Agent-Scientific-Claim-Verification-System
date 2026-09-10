"""Unit tests for MASCV FastAPI Backend endpoints."""

from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from ui.backend.main import app
from ui.backend.session_manager import get_session, set_session
from mascv.core.state import InvestigationState, ClaimInvestigationState
from mascv.models.paper import ResearchPaper, PaperMetadata
from mascv.models.claim import Claim, ClaimType
from mascv.models.argument import Argument
from mascv.models.verdict import Verdict, VerdictType, CriticFinding

client = TestClient(app)


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_get_active_sessions():
    response = client.get("/api/sessions")
    assert response.status_code == 200
    assert "sessions" in response.json()


def test_load_sample_paper():
    response = client.post("/api/upload/sample?sample_name=lora_2106.09685.pdf")
    assert response.status_code == 200
    data = response.json()
    assert "paper_id" in data
    assert "LORA" in data["title"]
    assert data["sections_count"] > 0


def test_load_nonexistent_sample_paper():
    response = client.post("/api/upload/sample?sample_name=non_existent_paper.pdf")
    assert response.status_code == 404


def test_get_claims_and_debate():
    # Setup mock session state
    paper = ResearchPaper(
        id="P-TEST",
        metadata=PaperMetadata(title="Test Paper", paper_id="P-TEST"),
        raw_text="Test paper text",
    )
    claim = Claim(
        id="C1",
        paper_id="P-TEST",
        subject="Efficiency",
        statement="Method X reduces memory by 3x.",
        claim_type=ClaimType.EFFICIENCY,
    )
    support_arg = Argument(
        agent_name="SupportAgent",
        claim_id="C1",
        stance="FOR",
        strength="Strong",
        conclusion="Evidence confirms 3x reduction.",
        premises=["VRAM was reduced by 3x on benchmarks."],
        cited_evidence_ids=["E-1"],
    )
    attack_arg = Argument(
        agent_name="AttackAgent",
        claim_id="C1",
        stance="AGAINST",
        strength="Moderate",
        conclusion="Vulnerable to task complexity bounds.",
        premises=["Capacity limits observed on complex reasoning."],
        cited_evidence_ids=["https://arxiv.org/abs/2401.00000"],
    )
    finding = CriticFinding(
        citation_valid=True,
        reasoning_sound=True,
        overgeneralization_detected=False,
        fair_comparison=True,
        critique_notes="Evaluation was direct.",
    )
    verdict = Verdict(
        claim_id="C1",
        verdict=VerdictType.SUPPORTED,
        confidence=0.92,
        critic_finding=finding,
        strongest_supporting_argument="Consistent VRAM reduction.",
        strongest_counterargument="Potential capacity constraints.",
        synthesis_summary="The claim is strongly supported.",
    )
    cstate = ClaimInvestigationState(
        claim=claim,
        support_argument=support_arg,
        attack_argument=attack_arg,
        verdict=verdict,
        is_finalized=True,
    )
    state = InvestigationState(
        paper=paper,
        claims={"C1": cstate},
        active_claim_id="C1",
    )
    set_session("P-TEST", state)

    # 1. Get claims
    resp = client.get("/api/claims/P-TEST")
    assert resp.status_code == 200
    claims = resp.json()
    assert len(claims) == 1
    assert claims[0]["claim_id"] == "C1"
    assert claims[0]["verdict"] == "Supported"
    assert claims[0]["confidence"] == 0.92

    # 2. Get debate
    resp = client.get("/api/debate/P-TEST/C1")
    assert resp.status_code == 200
    debate = resp.json()
    assert debate["claim_id"] == "C1"
    assert debate["support_argument"]["stance"] == "FOR"
    assert debate["attack_argument"]["stance"] == "AGAINST"

    # 3. Get report
    resp = client.get("/api/reports/P-TEST")
    assert resp.status_code == 200
    report = resp.json()
    assert report["paper_id"] == "P-TEST"
    assert report["total_claims"] == 1
    assert len(report["verdicts"]) == 1
    assert "Executive Summary" in report["executive_summary"]
