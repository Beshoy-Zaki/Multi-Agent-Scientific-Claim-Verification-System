"""End-to-End Live Integration Test: Verifies all agents collaborating on LoRA paper using Gemma 4."""

import os
import sys

# Ensure src is on python path
sys.path.insert(0, "src")

from dotenv import load_dotenv
load_dotenv()

from mascv.core.state import InvestigationState
from mascv.rag.parsers.pdf_parser import PDFParser
from mascv.agents.supervisor import SupervisorAgent
from mascv.agents.claim_analyst import ClaimAnalystAgent
from mascv.agents.paper_search import PaperSearchAgent
from mascv.agents.evidence_rag import EvidenceRAGAgent
from mascv.agents.support_agent import SupportAgent


def p(msg: str = ""):
    print(msg, flush=True)


def main():
    p("=" * 80)
    p("MASCV MULTI-AGENT PIPELINE INTEGRATION TEST (GEMMA 4)")
    p("=" * 80)

    # 1. Parse Paper
    pdf_path = os.path.join("data", "sample_inputs", "lora_2106.09685.pdf")
    p(f"\n[1] Ingesting PDF: {pdf_path}")
    parser = PDFParser()
    paper = parser.parse(pdf_path)
    p(f"    Paper Parsed: '{paper.metadata.title}' ({len(paper.sections)} sections)")

    state = InvestigationState(paper=paper)

    # 2. Supervisor Check
    p("\n[2] Initializing SupervisorAgent...")
    supervisor = SupervisorAgent()
    decision = supervisor.decide_next_step(state)
    p(f"    Supervisor Decision 1: Next agent is '{decision}'")

    # 3. Claim Analyst Agent
    p("\n[3] Running ClaimAnalystAgent with Gemma 4...")
    claim_analyst = ClaimAnalystAgent()
    state = claim_analyst.execute(state)
    p(f"    Extracted {len(state.claims)} claims:")
    for cid, cstate in state.claims.items():
        p(f"    - [{cid}] {cstate.claim.statement[:85]}...")

    # Select active claim C1
    active_cid = next(iter(state.claims))
    state.active_claim_id = active_cid
    active_claim = state.claims[active_cid].claim
    p(f"\n    Active Claim Selected: [{active_cid}] '{active_claim.statement}'")

    # 4. Supervisor Check
    decision = supervisor.decide_next_step(state)
    p(f"\n[4] Supervisor Decision 2: Next agent is '{decision}'")

    # 5. Paper Search Agent (with search grounding)
    p("\n[5] Running PaperSearchAgent with Gemma 4 Search Grounding...")
    search_agent = PaperSearchAgent()
    state = search_agent.execute(state)
    claim_state = state.claims[active_cid]
    p(f"    External papers found for [{active_cid}]: {len(claim_state.external_papers_found)}")
    for i, title in enumerate(claim_state.external_papers_found[:3], 1):
        p(f"    - Paper {i}: {title}")

    # 6. Evidence RAG Agent (chunking + hybrid retrieval + extraction)
    p("\n[6] Running EvidenceRAGAgent with HybridRetriever & Gemma 4...")
    rag_agent = EvidenceRAGAgent()
    state = rag_agent.execute(state)
    active_state = state.claims[active_cid]
    evidence_ids = active_state.get("evidence_bundle_ids", []) if isinstance(active_state, dict) else active_state.evidence_bundle_ids
    p(f"    Evidence Bundles Extracted: {len(evidence_ids)}")
    for eid in evidence_ids[:3]:
        eb = state.global_evidence_store.get(eid)
        if eb:
            rel = getattr(eb, "relationship", eb.get("relationship") if isinstance(eb, dict) else "")
            cnt = getattr(eb, "content", eb.get("content") if isinstance(eb, dict) else "")
            p(f"    - Bundle [{eid}]: ({rel}) {cnt[:80]}...")

    # 7. Support Agent (proponent argument construction)
    p("\n[7] Running SupportAgent with Gemma 4...")
    support_agent = SupportAgent()
    state = support_agent.execute(state)
    active_state = state.claims[active_cid]
    arg = active_state.get("support_argument") if isinstance(active_state, dict) else active_state.support_argument
    if arg:
        stance = getattr(arg, "stance", arg.get("stance") if isinstance(arg, dict) else "")
        strength = getattr(arg, "strength", arg.get("strength") if isinstance(arg, dict) else "")
        conclusion = getattr(arg, "conclusion", arg.get("conclusion") if isinstance(arg, dict) else "")
        premises = getattr(arg, "premises", arg.get("premises", []) if isinstance(arg, dict) else [])
        cited = getattr(arg, "cited_evidence_ids", arg.get("cited_evidence_ids", []) if isinstance(arg, dict) else [])
        p(f"    Support Argument Formed:")
        p(f"    - Stance:     {stance}")
        p(f"    - Strength:   {strength}")
        p(f"    - Conclusion: {conclusion}")
        p(f"    - Premises:   {len(premises)}")
        p(f"    - Cited IDs:  {cited}")

    p("\n" + "=" * 80)
    p("SUCCESS: ALL 5 INTEGRATED AGENTS EXECUTED TOGETHER CLEANLY!")
    p("=" * 80)


if __name__ == "__main__":
    main()

