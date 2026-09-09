"""End-to-End Live Integration Test: Verifies all agents collaborating on LoRA paper using Gemma 4."""

import os
import sys

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
from mascv.agents.attack_agent import AttackAgent
from mascv.agents.critic_agent import CriticAgent


def p(msg: str = ""):
    print(msg, flush=True)


def main():
    p("=" * 80)
    p("MASCV MULTI-AGENT PIPELINE INTEGRATION TEST (GEMMA 4 - ALL 7 AGENTS)")
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
    p("\n[7] Running SupportAgent (Agent 5) with Gemma 4...")
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
        p(f"    - Conclusion: {conclusion[:100]}...")
        p(f"    - Premises:   {len(premises)}")
        p(f"    - Cited IDs:  {cited}")

    # 8. Attack Agent (adversarial counterargument construction)
    p("\n[8] Running AttackAgent (Agent 6) with Gemma 4 & Scientific Search...")
    attack_agent = AttackAgent()
    state = attack_agent.execute(state)
    active_state = state.claims[active_cid]
    attack_arg = active_state.get("attack_argument") if isinstance(active_state, dict) else active_state.attack_argument
    if attack_arg:
        astance = getattr(attack_arg, "stance", attack_arg.get("stance") if isinstance(attack_arg, dict) else "")
        astrength = getattr(attack_arg, "strength", attack_arg.get("strength") if isinstance(attack_arg, dict) else "")
        aconclusion = getattr(attack_arg, "conclusion", attack_arg.get("conclusion") if isinstance(attack_arg, dict) else "")
        apremises = getattr(attack_arg, "premises", attack_arg.get("premises", []) if isinstance(attack_arg, dict) else [])
        alimitations = getattr(attack_arg, "identified_limitations", attack_arg.get("identified_limitations", []) if isinstance(attack_arg, dict) else [])
        acited = getattr(attack_arg, "cited_evidence_ids", attack_arg.get("cited_evidence_ids", []) if isinstance(attack_arg, dict) else [])
        p(f"    Attack Counterargument Formed:")
        p(f"    - Stance:       {astance}")
        p(f"    - Strength:     {astrength}")
        p(f"    - Conclusion:   {aconclusion[:100]}...")
        p(f"    - Attack Points:{len(apremises)}")
        p(f"    - Limitations:  {len(alimitations)}")
        p(f"    - Counter-URLs: {acited}")

    # 9. Critic Agent (dialectical adjudication & verdict synthesis)
    p("\n[9] Running CriticAgent (Agent 7) with Gemma 4...")
    critic_agent = CriticAgent()
    state = critic_agent.execute(state)
    active_state = state.claims[active_cid]
    verdict = active_state.get("verdict") if isinstance(active_state, dict) else active_state.verdict
    if verdict:
        v_type = getattr(verdict, "verdict", verdict.get("verdict") if isinstance(verdict, dict) else "")
        v_conf = getattr(verdict, "confidence", verdict.get("confidence") if isinstance(verdict, dict) else 0.0)
        v_summary = getattr(verdict, "synthesis_summary", verdict.get("synthesis_summary") if isinstance(verdict, dict) else "")
        cf = getattr(verdict, "critic_finding", verdict.get("critic_finding") if isinstance(verdict, dict) else None)
        p(f"    Final Synthesized Verdict:")
        p(f"    - Verdict:         {v_type.value if hasattr(v_type, 'value') else v_type}")
        p(f"    - Confidence:      {v_conf:.2f}")
        p(f"    - Synthesis:       {v_summary[:120]}...")
        if cf:
            notes = getattr(cf, "critique_notes", cf.get("critique_notes") if isinstance(cf, dict) else "")
            p(f"    - Critique Notes:  {notes}")

    # 10. Supervisor Finalization
    p("\n[10] Finalizing Multi-Agent Investigation with Supervisor...")
    final_decision = supervisor.decide_next_step(state)
    p(f"    Supervisor Final Decision: '{final_decision}'")
    is_final = active_state.get("is_finalized") if isinstance(active_state, dict) else active_state.is_finalized
    p(f"    Claim [{active_cid}] Finalized: {is_final}")

    state = supervisor.execute(state)
    exec_summary = supervisor.generate_executive_summary(state)
    p(f"\n[11] Multi-Agent Investigation Executive Summary:\n{exec_summary}")

    p("\n" + "=" * 80)
    p("SUCCESS: COMPLETE 7-AGENT PIPELINE EXECUTED END-TO-END WITH GEMMA 4!")
    p("=" * 80)


if __name__ == "__main__":
    main()


