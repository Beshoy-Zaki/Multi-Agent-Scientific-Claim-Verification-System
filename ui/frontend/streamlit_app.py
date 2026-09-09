"""Multi-Agent Scientific Claim Verification System (MASCV) - Research Dashboard."""

import os
import sys
import json
from datetime import datetime
from typing import Dict, Any, Optional

# Ensure src is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from mascv.core.state import InvestigationState, ClaimInvestigationState
from mascv.rag.parsers.pdf_parser import PDFParser
from mascv.agents.supervisor import SupervisorAgent
from mascv.agents.claim_analyst import ClaimAnalystAgent
from mascv.agents.paper_search import PaperSearchAgent
from mascv.agents.evidence_rag import EvidenceRAGAgent
from mascv.agents.support_agent import SupportAgent
from mascv.agents.attack_agent import AttackAgent
from mascv.agents.critic_agent import CriticAgent
from mascv.models.verdict import VerdictType

# -----------------------------------------------------------------------------
# Streamlit Configuration & Custom Lively CSS Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MASCV - Multi-Agent Scientific Claim Verification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* Gradient Header Banner */
.mascv-hero {
    background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 50%, #EC4899 100%);
    padding: 24px 32px;
    border-radius: 16px;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.4);
}
.mascv-hero h1 {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0;
    color: #FFFFFF !important;
}
.mascv-hero p {
    font-size: 1.05rem;
    margin-top: 6px;
    opacity: 0.92;
    color: #F8FAFC !important;
}

/* Agent Pill Badges */
.agent-pill {
    display: inline-flex;
    align-items: center;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-right: 8px;
}
.agent-supervisor { background-color: #EDE9FE; color: #6D28D9; border: 1px solid #DDD6FE; }
.agent-analyst    { background-color: #E0F2FE; color: #0369A1; border: 1px solid #BAE6FD; }
.agent-search     { background-color: #CCFBF1; color: #0F766E; border: 1px solid #99F6E4; }
.agent-rag        { background-color: #F3E8FF; color: #7E22CE; border: 1px solid #E9D5FF; }
.agent-support    { background-color: #D1FAE5; color: #047857; border: 1px solid #A7F3D0; }
.agent-attack     { background-color: #FFE4E6; color: #BE123C; border: 1px solid #FECDD3; }
.agent-critic     { background-color: #FEF3C7; color: #B45309; border: 1px solid #FDE68A; }

/* Verdict Hero Banners */
.verdict-card {
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
    border-left: 8px solid;
}
.verdict-supported {
    background-color: #ECFDF5;
    border-color: #10B981;
    color: #065F46;
}
.verdict-partially {
    background-color: #FFFBEB;
    border-color: #F59E0B;
    color: #92400E;
}
.verdict-unsupported {
    background-color: #FEF2F2;
    border-color: #EF4444;
    color: #991B1B;
}
.verdict-inconclusive {
    background-color: #F8FAFC;
    border-color: #64748B;
    color: #334155;
}

/* Side-by-Side Debate Cards */
.debate-box {
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}
.support-box {
    background: linear-gradient(180deg, #F0FDF4 0%, #FFFFFF 100%);
    border: 2px solid #86EFAC;
}
.attack-box {
    background: linear-gradient(180deg, #FFF1F2 0%, #FFFFFF 100%);
    border: 2px solid #FDA4AF;
}

/* Claim Cards */
.claim-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 12px;
    transition: all 0.2s ease;
}
.claim-card:hover {
    border-color: #6366F1;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.1);
}

/* Tag Pills */
.tag-pill {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 6px;
    background-color: #F1F5F9;
    color: #475569;
}

/* Literature Search Card */
.search-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-left: 4px solid #0284C7;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
    transition: all 0.2s ease;
}
.search-card:hover {
    border-left-color: #0369A1;
    box-shadow: 0 4px 12px rgba(2, 132, 199, 0.12);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "state" not in st.session_state:
    st.session_state.state = None
if "active_claim_id" not in st.session_state:
    st.session_state.active_claim_id = None
if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False
if "pipeline_step" not in st.session_state:
    st.session_state.pipeline_step = 0
if "execution_logs" not in st.session_state:
    st.session_state.execution_logs = []

def add_log(agent: str, message: str):
    time_str = datetime.now().strftime("%H:%M:%S")
    st.session_state.execution_logs.append({"time": time_str, "agent": agent, "message": message})


# -----------------------------------------------------------------------------
# Top Hero Banner
# -----------------------------------------------------------------------------
st.markdown("""
<div class="mascv-hero">
    <h1>🔬 Multi-Agent Scientific Claim Verification System</h1>
    <p>Evidence-Grounded Dialectic Analysis & Automated Peer Critique powered by Google Gemma 4</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar Configuration & Progress Tracker
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Investigation Setup")
    
    # Check Google API Key Status
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key:
        st.success("🟢 Google Gemma 4 API Connected")
    else:
        st.error("🔴 GOOGLE_API_KEY Missing in Environment")
        api_input = st.text_input("Enter Google API Key:", type="password")
        if api_input:
            os.environ["GOOGLE_API_KEY"] = api_input
            os.environ["GEMINI_API_KEY"] = api_input
            st.rerun()

    st.markdown("---")
    st.subheader("📄 Target Publication")

    paper_source = st.radio(
        "Choose Paper Source:",
        ["Packaged Benchmark Paper", "Upload Custom PDF"],
        index=0,
    )

    pdf_to_load = None
    if paper_source == "Packaged Benchmark Paper":
        samples = ["lora_2106.09685.pdf"]
        selected_sample = st.selectbox("Select Benchmark Sample:", samples)
        pdf_to_load = os.path.join("data", "sample_inputs", selected_sample)
    else:
        uploaded = st.file_uploader("Upload PDF file", type=["pdf"])
        if uploaded:
            temp_path = os.path.join("data", "raw_papers", uploaded.name)
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded.getbuffer())
            pdf_to_load = temp_path

    st.markdown("---")
    st.subheader("🚀 Investigation Pipeline")

    # One-Click Run Button
    run_disabled = pdf_to_load is None or not os.path.exists(pdf_to_load)
    if st.button("▶️ Run Full Multi-Agent Pipeline", type="primary", use_container_width=True, disabled=run_disabled):
        st.session_state.pipeline_running = True
        st.session_state.target_pdf = pdf_to_load

    if st.button("🔄 Reset Investigation", use_container_width=True):
        st.session_state.state = None
        st.session_state.active_claim_id = None
        st.session_state.pipeline_step = 0
        st.session_state.execution_logs = []
        st.rerun()

    # Progress Indicator
    st.markdown("---")
    st.subheader("📊 Pipeline Stage")
    steps = [
        "1. Paper Ingestion",
        "2. Claims Extraction",
        "3. Grounded Search",
        "4. Evidence Extraction",
        "5. Support Case",
        "6. Adversarial Attack",
        "7. Critic Verdict",
        "8. Executive Summary",
    ]
    cur_step = st.session_state.pipeline_step
    progress_val = min(1.0, max(0.0, cur_step / len(steps)))
    st.progress(progress_val)
    st.caption(f"Current Phase: **{steps[min(cur_step, len(steps)-1)]}** ({int(progress_val*100)}%)")


# -----------------------------------------------------------------------------
# Multi-Agent Execution Handler (with Real-Time Live Streaming)
# -----------------------------------------------------------------------------
if st.session_state.get("pipeline_running", False):
    pdf_path = st.session_state.get("target_pdf")
    
    with st.status("🤖 Executing MASCV Multi-Agent Pipeline in Real Time...", expanded=True) as status_box:
        try:
            live_progress = st.progress(0.0)

            # ---------------------------------------------------------
            # 1. Paper Ingestion
            # ---------------------------------------------------------
            st.markdown("### 📄 Step 1/8: Ingesting & Structuring Paper (`PDFParser`)")
            with st.spinner("Extracting text and structural sections from PDF..."):
                parser = PDFParser()
                paper = parser.parse(pdf_path)
                state = InvestigationState(paper=paper)
                st.session_state.pipeline_step = 1
                add_log("PDFParser", f"Parsed '{paper.metadata.title}' into {len(paper.sections)} sections.")

            live_progress.progress(1 / 8)
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(f"**Title:** {paper.metadata.title or 'Untitled Publication'}")
                if paper.metadata.authors:
                    st.caption(f"Authors: {', '.join(paper.metadata.authors)}")
            with col2:
                st.metric("Sections Parsed", len(paper.sections))
            with col3:
                words_count = len(paper.raw_text.split()) if paper.raw_text else 0
                st.metric("Word Count", f"{words_count:,}")

            with st.expander(f"📑 Preview Parsed Sections ({len(paper.sections)} detected)", expanded=False):
                for sec in paper.sections[:6]:
                    st_title = getattr(sec, "title", "Section") or "Section"
                    st_text = getattr(sec, "content", "") or ""
                    st.markdown(f"• **{st_title}** ({len(st_text.split())} words)")

            st.markdown("---")

            # ---------------------------------------------------------
            # 2. Extract Claims
            # ---------------------------------------------------------
            st.markdown("### 📋 Step 2/8: Formalizing Testable Claims (`ClaimAnalystAgent` via Gemma 4)")
            with st.spinner("Deconstructing publication into formalized scientific propositions..."):
                analyst = ClaimAnalystAgent()
                state = analyst.execute(state)
                st.session_state.pipeline_step = 2
                add_log("ClaimAnalystAgent", f"Formulated {len(state.claims)} testable scientific claims.")

            live_progress.progress(2 / 8)
            st.success(f"🎯 **ClaimAnalystAgent** formalized **{len(state.claims)} testable scientific claims**:")

            for cid, cstate in state.claims.items():
                claim = cstate.claim
                ctype = claim.claim_type.value if hasattr(claim.claim_type, "value") else str(claim.claim_type)
                st.markdown(f"""
                <div class="claim-card" style="border-left: 5px solid #6366F1; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 700; color: #4338CA;">[{cid}] {claim.subject or 'Scientific Proposition'}</span>
                        <span class="tag-pill" style="background-color: #E0E7FF; color: #3730A3;">{ctype.upper()}</span>
                    </div>
                    <p style="margin: 6px 0 4px 0; color: #1E293B;"><b>Statement:</b> {claim.statement}</p>
                </div>
                """, unsafe_allow_html=True)
                if claim.benchmarks or claim.metrics:
                    tags = " ".join([f"`📊 {b}`" for b in claim.benchmarks] + [f"`📈 {m}`" for m in claim.metrics])
                    st.caption(f"Grounding Targets: {tags}")

            # Select primary active claim
            active_id = state.active_claim_id or (list(state.claims.keys())[0] if state.claims else None)
            st.session_state.active_claim_id = active_id
            state.active_claim_id = active_id

            st.markdown("---")

            if active_id:
                # ---------------------------------------------------------
                # 3. Paper Search Agent
                # ---------------------------------------------------------
                active_claim = state.claims[active_id].claim
                st.markdown(f"### 🌐 Step 3/8: Literature Discovery via Google Search Grounding (`PaperSearchAgent`)")
                st.info(f"Targeting Claim **[{active_id}]**: *\"{active_claim.statement[:120]}...\"*")

                with st.spinner("Executing Google Grounded Search for peer publications and replications..."):
                    searcher = PaperSearchAgent()
                    state = searcher.execute(state)
                    st.session_state.pipeline_step = 3
                    discovered_meta = getattr(state.claims[active_id], "discovered_papers_metadata", [])
                    discovered_titles = getattr(state.claims[active_id], "external_papers_found", [])
                    total_discovered = len(discovered_meta) if discovered_meta else len(discovered_titles)
                    add_log("PaperSearchAgent", f"Retrieved {total_discovered} external citations/papers for claim [{active_id}].")

                live_progress.progress(3 / 8)
                st.success(f"🌐 Discovered **{total_discovered} relevant academic literature sources**:")

                if discovered_meta:
                    for p in discovered_meta:
                        p_url = p.url or f"https://scholar.google.com/scholar?q={p.title.replace(' ', '+')}"
                        p_score = int((p.relevance_score or 0.85) * 100) if p.relevance_score else 85
                        p_rel = p.relationship or "RELEVANT"
                        st.markdown(f"""
                        <div class="search-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <a href="{p_url}" target="_blank" style="font-weight: 700; color: #0284C7; text-decoration: none;">📄 {p.title}</a>
                                <span class="tag-pill" style="background: #E0F2FE; color: #0369A1;">{p_rel} • {p_score}% RELEVANCE</span>
                            </div>
                            <div style="font-size: 0.85rem; color: #64748B; margin-top: 4px;">
                                Authors: {', '.join(p.authors[:3]) if p.authors else 'Academic Authors'} ({p.year or 'N/A'}) • Venue: {p.venue or 'Repository'}
                            </div>
                            {f'<div style="font-size: 0.85rem; color: #334155; margin-top: 4px; font-style: italic;"><b>Findings:</b> {p.relevance_rationale or p.key_findings}</div>' if (p.relevance_rationale or p.key_findings) else ''}
                        </div>
                        """, unsafe_allow_html=True)
                elif discovered_titles:
                    for t in discovered_titles:
                        st.markdown(f"- 📄 **{t}**")

                st.markdown("---")

                # ---------------------------------------------------------
                # 4. Evidence RAG Agent
                # ---------------------------------------------------------
                st.markdown(f"### 📚 Step 4/8: Grounded Evidence Extraction & Bundling (`EvidenceRAGAgent`)")
                with st.spinner("Extracting & verifying evidence passages across target paper and literature..."):
                    rag = EvidenceRAGAgent()
                    state = rag.execute(state)
                    st.session_state.pipeline_step = 4
                    evidence_items = list(state.global_evidence_store.values())
                    add_log("EvidenceRAGAgent", f"Bundled {len(evidence_items)} evidence passages in global store.")

                live_progress.progress(4 / 8)
                st.success(f"📚 Extracted & bundled **{len(evidence_items)} claim-aware evidence units**:")

                with st.expander(f"🔍 Inspect Grounded Evidence Bundles ({len(evidence_items)} total)", expanded=True):
                    for b in evidence_items[:5]:
                        b_rel = b.relationship.value if hasattr(b.relationship, "value") else str(b.relationship)
                        b_color = "#10B981" if b_rel in ["SUPPORTS", "REPLICATES"] else ("#EF4444" if b_rel == "CONTRADICTS" else "#F59E0B")
                        st.markdown(f"""
                        <div style="border-left: 3px solid {b_color}; padding-left: 10px; margin-bottom: 10px;">
                            <span class="tag-pill" style="background-color: {b_color}20; color: {b_color}; font-weight: bold;">{b_rel}</span>
                            <span style="font-size: 0.85rem; color: #64748B;">Source: <b>{b.source_title}</b> ({b.location or 'Document'})</span>
                            <p style="margin: 4px 0 0 0; font-size: 0.9rem; color: #1E293B;">"{b.content[:240]}..."</p>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("---")

                # ---------------------------------------------------------
                # 5. Support Agent
                # ---------------------------------------------------------
                st.markdown(f"### 🛡️ Step 5/8: Affirmative Case Construction (`SupportAgent`)")
                with st.spinner("Synthesizing grounded affirmative argument with premises and evidence citations..."):
                    support = SupportAgent()
                    state = support.execute(state)
                    st.session_state.pipeline_step = 5
                    sup_arg = state.claims[active_id].support_argument
                    add_log("SupportAgent", f"Constructed affirmative argument for [{active_id}].")

                live_progress.progress(5 / 8)
                if sup_arg:
                    st.markdown(f"""
                    <div class="debate-box support-box">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span class="agent-pill agent-support">🛡️ SUPPORT AGENT</span>
                            <span style="font-weight: 700; color: #047857;">STANCE: FOR • STRENGTH: {sup_arg.strength.upper()}</span>
                        </div>
                        <p style="font-weight: 600; color: #065F46; margin-bottom: 6px;">Conclusion: {sup_arg.conclusion}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    with st.expander("View Support Affirmative Premises", expanded=False):
                        for p in sup_arg.premises:
                            st.write(f"• {p}")
                        if sup_arg.cited_evidence_ids:
                            st.caption(f"Cited Evidence IDs: {', '.join(sup_arg.cited_evidence_ids)}")

                st.markdown("---")

                # ---------------------------------------------------------
                # 6. Attack Agent
                # ---------------------------------------------------------
                st.markdown(f"### ⚔️ Step 6/8: Adversarial Attack & Boundary Testing (`AttackAgent`)")
                with st.spinner("Searching counter-evidence and probing methodological vulnerabilities..."):
                    attack = AttackAgent()
                    state = attack.execute(state)
                    st.session_state.pipeline_step = 6
                    atk_arg = state.claims[active_id].attack_argument
                    add_log("AttackAgent", f"Constructed adversarial counter-case for [{active_id}].")

                live_progress.progress(6 / 8)
                if atk_arg:
                    st.markdown(f"""
                    <div class="debate-box attack-box">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span class="agent-pill agent-attack">⚔️ ATTACK AGENT</span>
                            <span style="font-weight: 700; color: #BE123C;">STANCE: AGAINST • STRENGTH: {atk_arg.strength.upper()}</span>
                        </div>
                        <p style="font-weight: 600; color: #9F1239; margin-bottom: 6px;">Counter-Conclusion: {atk_arg.conclusion}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    with st.expander("View Attack Counter-Premises & Vulnerabilities", expanded=False):
                        for p in atk_arg.premises:
                            st.write(f"• {p}")
                        if atk_arg.identified_limitations:
                            st.write("**Identified Limitations:**")
                            for lim in atk_arg.identified_limitations:
                                st.caption(f"⚠️ {lim}")
                        if atk_arg.cited_evidence_ids:
                            st.write("**External Counter-Evidence Links:**")
                            for link in atk_arg.cited_evidence_ids:
                                st.markdown(f"- [{link}]({link})")

                st.markdown("---")

                # ---------------------------------------------------------
                # 7. Critic Agent
                # ---------------------------------------------------------
                st.markdown(f"### ⚖️ Step 7/8: Adjudicating Debate & 4-Point Peer Audit (`CriticAgent`)")
                with st.spinner("Impartially evaluating debate, verifying citations, and assessing reasoning soundness..."):
                    critic = CriticAgent()
                    state = critic.execute(state)
                    st.session_state.pipeline_step = 7
                    verdict_obj = state.claims[active_id].verdict
                    add_log("CriticAgent", f"Synthesized verdict for claim [{active_id}].")

                live_progress.progress(7 / 8)
                if verdict_obj:
                    v_type = verdict_obj.verdict.value if hasattr(verdict_obj.verdict, "value") else str(verdict_obj.verdict)
                    conf = int((verdict_obj.confidence or 0.0) * 100)
                    finding = verdict_obj.critic_finding

                    st.markdown(f"""
                    <div style="background: #F8FAFC; border: 2px solid #6366F1; border-radius: 12px; padding: 14px 18px; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h3 style="margin: 0; color: #4338CA;">⚖️ Final Verdict: {v_type}</h3>
                            <span style="font-size: 1.2rem; font-weight: 800; color: #4338CA;">{conf}% Confidence</span>
                        </div>
                        <p style="margin: 8px 0 0 0; color: #334155; font-size: 0.95rem;"><b>Scientific Synthesis:</b> {verdict_obj.synthesis_summary}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    if finding:
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            st.metric("Citation Grounding", "Verified ✅" if getattr(finding, "citation_valid", False) else "Unverified ❌")
                        with c2:
                            st.metric("Reasoning Soundness", "Sound ✅" if getattr(finding, "reasoning_sound", False) else "Flawed ❌")
                        with c3:
                            st.metric("Overgeneralization", "Clean ✅" if not getattr(finding, "overgeneralization_detected", False) else "Detected ⚠️")
                        with c4:
                            st.metric("Comparative Parity", "Fair ✅" if getattr(finding, "fair_comparison", False) else "Asymmetric ❌")

                st.markdown("---")

            # ---------------------------------------------------------
            # 8. Supervisor Executive Summary
            # ---------------------------------------------------------
            st.markdown("### 🧭 Step 8/8: Executive Summary Synthesis (`SupervisorAgent`)")
            with st.spinner("Synthesizing multi-agent executive assessment report..."):
                supervisor = SupervisorAgent()
                summary = supervisor.generate_executive_summary(state)
                st.session_state.executive_summary = summary
                st.session_state.pipeline_step = 8
                add_log("SupervisorAgent", "Executive summary synthesized.")

            live_progress.progress(1.0)
            st.success("🧭 **SupervisorAgent** synthesized full scientific assessment report.")
            with st.expander("📄 Preview Synthesized Executive Summary", expanded=False):
                st.markdown(summary)

            # Completion & state persistence
            st.session_state.state = state
            st.session_state.pipeline_running = False
            st.session_state.pipeline_completed = True
            status_box.update(label="🎉 Multi-Agent Pipeline Execution Succeeded!", state="complete", expanded=True)
            st.balloons()

        except Exception as exc:
            status_box.update(label=f"❌ Pipeline Failed: {exc}", state="error", expanded=True)
            st.session_state.pipeline_running = False
            st.error(f"Error during execution: {exc}")


# -----------------------------------------------------------------------------
# Main Dashboard Multi-View Tabs (with Live Activity Stream)
# -----------------------------------------------------------------------------
tab_stream, tab_paper, tab_claims, tab_evidence, tab_debate, tab_verdict = st.tabs([
    "⚡ Live Activity Stream",
    "📄 Paper Overview",
    "🎯 Extracted Claims",
    "🔍 Evidence Explorer",
    "⚔️ Dialectic Debate Arena",
    "⚖️ Verdict & Scientific Report",
])

state = st.session_state.state

# -----------------------------------------------------------------------------
# TAB 0: Live Activity Stream
# -----------------------------------------------------------------------------
with tab_stream:
    st.header("⚡ Real-Time Multi-Agent Activity Stream")
    if state and state.claims:
        st.write("Complete chronological trace of all 8 multi-agent verification phases:")

        # Summary Metric Bar
        active_id = st.session_state.active_claim_id or (list(state.claims.keys())[0] if state.claims else None)
        s_cstate = state.claims.get(active_id) if active_id else None
        s_verdict = s_cstate.verdict if s_cstate and hasattr(s_cstate, "verdict") else None
        v_str = s_verdict.verdict.value if s_verdict and hasattr(s_verdict.verdict, "value") else (str(s_verdict.verdict) if s_verdict else "Pending")
        conf_str = f"{int(s_verdict.confidence * 100)}%" if s_verdict and s_verdict.confidence else "N/A"

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Total Claims", len(state.claims))
        with m2:
            st.metric("Active Claim", active_id or "None")
        with m3:
            papers_found_count = len(getattr(s_cstate, "discovered_papers_metadata", [])) or len(getattr(s_cstate, "external_papers_found", [])) if s_cstate else 0
            st.metric("Discovered Sources", papers_found_count)
        with m4:
            st.metric("Evidence Bundles", len(state.global_evidence_store))
        with m5:
            st.metric("Active Verdict", f"{v_str} ({conf_str})")

        st.markdown("---")

        # 1. Ingested Paper Summary
        with st.expander("📄 Phase 1: Ingested Paper & Structure (`PDFParser`)", expanded=True):
            if state.paper:
                p_meta = state.paper.metadata
                st.markdown(f"**Publication Title:** {p_meta.title or 'Untitled Paper'}")
                if p_meta.authors:
                    st.caption(f"Authors: {', '.join(p_meta.authors)}")
                st.write(f"Parsed **{len(state.paper.sections)} sections** across **{len(state.paper.raw_text.split()):,} words**.")

        # 2. Extracted Claims
        with st.expander(f"🎯 Phase 2: Formalized Scientific Claims ({len(state.claims)} extracted via `ClaimAnalystAgent`)", expanded=True):
            for cid, cstate in state.claims.items():
                c = cstate.claim
                ctype = c.claim_type.value if hasattr(c.claim_type, "value") else str(c.claim_type)
                is_act = (cid == active_id)
                card_bg = "#EEF2FF" if is_act else "#F8FAFC"
                st.markdown(f"""
                <div style="background: {card_bg}; border: 1px solid #CBD5E1; border-left: 4px solid {'#4F46E5' if is_act else '#94A3B8'}; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between;">
                        <b>[{cid}] {c.subject or 'Scientific Proposition'}</b>
                        <span class="tag-pill" style="background: #E0E7FF; color: #3730A3;">{ctype.upper()}</span>
                    </div>
                    <p style="margin: 4px 0; color: #1E293B;"><b>Statement:</b> {c.statement}</p>
                </div>
                """, unsafe_allow_html=True)

        # 3. Discovered Literature
        if s_cstate:
            disc_meta = getattr(s_cstate, "discovered_papers_metadata", [])
            disc_titles = getattr(s_cstate, "external_papers_found", [])
            with st.expander(f"🌐 Phase 3: External Literature via Google Search Grounding ({len(disc_meta) or len(disc_titles)} discovered for [{active_id}])", expanded=True):
                if disc_meta:
                    for p in disc_meta:
                        p_url = p.url or f"https://scholar.google.com/scholar?q={p.title.replace(' ', '+')}"
                        p_score = int((p.relevance_score or 0.85) * 100) if p.relevance_score else 85
                        p_rel = p.relationship or "RELEVANT"
                        st.markdown(f"""
                        <div class="search-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <a href="{p_url}" target="_blank" style="font-weight: 700; color: #0284C7; text-decoration: none;">📄 {p.title}</a>
                                <span class="tag-pill" style="background: #E0F2FE; color: #0369A1;">{p_rel} • {p_score}% MATCH</span>
                            </div>
                            <div style="font-size: 0.85rem; color: #64748B; margin-top: 4px;">
                                Authors: {', '.join(p.authors[:3]) if p.authors else 'Academic Authors'} ({p.year or 'N/A'}) • Venue: {p.venue or 'Academic Repository'}
                            </div>
                            {f'<div style="font-size: 0.85rem; color: #334155; margin-top: 4px; font-style: italic;"><b>Findings:</b> {p.relevance_rationale or p.key_findings}</div>' if (p.relevance_rationale or p.key_findings) else ''}
                        </div>
                        """, unsafe_allow_html=True)
                elif disc_titles:
                    for t in disc_titles:
                        st.markdown(f"- 📄 **{t}**")

        # 4. Evidence Extraction
        with st.expander(f"📚 Phase 4: Grounded Evidence Store ({len(state.global_evidence_store)} bundles extracted via `EvidenceRAGAgent`)", expanded=True):
            for b in list(state.global_evidence_store.values())[:6]:
                b_rel = b.relationship.value if hasattr(b.relationship, "value") else str(b.relationship)
                b_color = "#10B981" if b_rel in ["SUPPORTS", "REPLICATES"] else ("#EF4444" if b_rel == "CONTRADICTS" else "#F59E0B")
                st.markdown(f"""
                <div style="border-left: 3px solid {b_color}; padding-left: 10px; margin-bottom: 8px;">
                    <span class="tag-pill" style="background-color: {b_color}20; color: {b_color}; font-weight: bold;">{b_rel}</span>
                    <span style="font-size: 0.85rem; color: #64748B;">Source: <b>{b.source_title}</b> ({b.location or 'Document'})</span>
                    <p style="margin: 2px 0 0 0; font-size: 0.88rem; color: #1E293B;">"{b.content[:220]}..."</p>
                </div>
                """, unsafe_allow_html=True)

        # 5 & 6. Dialectic Debate
        if s_cstate:
            with st.expander("⚔️ Phases 5 & 6: Dialectic Debate Arena (`SupportAgent` vs `AttackAgent`)", expanded=True):
                dcol1, dcol2 = st.columns(2)
                with dcol1:
                    sup = s_cstate.support_argument
                    if sup:
                        st.markdown(f"""
                        <div class="debate-box support-box">
                            <span class="agent-pill agent-support">🛡️ SUPPORT AGENT</span>
                            <span style="font-weight: 700; color: #047857;">STANCE: FOR ({sup.strength})</span>
                            <p style="margin-top: 6px; font-weight: 600;">{sup.conclusion}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        for p in sup.premises:
                            st.caption(f"• {p}")
                with dcol2:
                    atk = s_cstate.attack_argument
                    if atk:
                        st.markdown(f"""
                        <div class="debate-box attack-box">
                            <span class="agent-pill agent-attack">⚔️ ATTACK AGENT</span>
                            <span style="font-weight: 700; color: #BE123C;">STANCE: AGAINST ({atk.strength})</span>
                            <p style="margin-top: 6px; font-weight: 600;">{atk.conclusion}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        for p in atk.premises:
                            st.caption(f"• {p}")

        # 7. Critic Verdict
        if s_verdict:
            with st.expander("⚖️ Phase 7: Critic Agent Adjudication & 4-Point Peer Audit", expanded=True):
                st.markdown(f"**Final Synthesized Verdict:** `{v_str}` ({conf_str} Confidence)")
                st.info(f"**Scientific Synthesis:** {s_verdict.synthesis_summary}")
                f_obj = s_verdict.critic_finding
                if f_obj:
                    qc1, qc2, qc3, qc4 = st.columns(4)
                    with qc1:
                        st.metric("Grounding", "Verified ✅" if getattr(f_obj, "citation_valid", False) else "Unverified ❌")
                    with qc2:
                        st.metric("Soundness", "Sound ✅" if getattr(f_obj, "reasoning_sound", False) else "Flawed ❌")
                    with qc3:
                        st.metric("Overgeneralization", "Clean ✅" if not getattr(f_obj, "overgeneralization_detected", False) else "Detected ⚠️")
                    with qc4:
                        st.metric("Parity", "Fair ✅" if getattr(f_obj, "fair_comparison", False) else "Asymmetric ❌")

        # 8. Executive Summary
        exec_text = st.session_state.get("executive_summary")
        if exec_text:
            with st.expander("🧭 Phase 8: Executive Scientific Assessment Summary (`SupervisorAgent`)", expanded=True):
                st.markdown(exec_text)

    else:
        st.info("👋 No multi-agent activity recorded yet. In the sidebar, select a paper and click **▶️ Run Full Multi-Agent Pipeline** to watch all agents execute live!")


state = st.session_state.state

# -----------------------------------------------------------------------------
# TAB 1: Paper Overview
# -----------------------------------------------------------------------------
with tab_paper:
    st.header("📄 Ingested Paper Overview")
    if state and state.paper:
        paper = state.paper
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.subheader(paper.metadata.title or "Untitled Paper")
            if paper.metadata.authors:
                st.write(f"**Authors:** {', '.join(paper.metadata.authors)}")
        with col2:
            st.metric("Sections Parsed", len(paper.sections))
        with col3:
            total_words = len(paper.raw_text.split()) if paper.raw_text else 0
            st.metric("Word Count", f"{total_words:,}")

        if paper.metadata.abstract:
            st.info(f"**Abstract:**\n\n{paper.metadata.abstract}")

        st.subheader("📑 Paper Section Browser")
        with st.expander("Explore Document Sections", expanded=False):
            for sec in paper.sections:
                sec_title = getattr(sec, "title", getattr(sec, "heading", "Section")) or "Section"
                sec_content = getattr(sec, "content", getattr(sec, "text", "")) or ""
                word_count = len(sec_content.split()) if sec_content else 0
                st.markdown(f"**{sec_title}** ({word_count} words)")
                if sec_content:
                    st.text(sec_content[:300] + ("..." if len(sec_content) > 300 else ""))
                st.markdown("---")
    else:
        st.info("👋 No publication loaded yet. Click **▶️ Run Full Multi-Agent Pipeline** in the sidebar to begin!")


# -----------------------------------------------------------------------------
# TAB 2: Extracted Claims
# -----------------------------------------------------------------------------
with tab_claims:
    st.header("🎯 Formalized Scientific Claims")
    if state and state.claims:
        st.write(f"The `ClaimAnalystAgent` identified and categorized **{len(state.claims)} testable claims**:")

        # Summary Metric Row
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        types_count = {}
        for c in state.claims.values():
            ctype = c.claim.claim_type.value if hasattr(c.claim.claim_type, "value") else str(c.claim.claim_type)
            types_count[ctype] = types_count.get(ctype, 0) + 1

        with mcol1:
            st.metric("Total Claims", len(state.claims))
        with mcol2:
            st.metric("Efficiency Claims", types_count.get("efficiency", 0))
        with mcol3:
            st.metric("Performance Claims", types_count.get("performance", 0))
        with mcol4:
            st.metric("Active Inspected Claim", st.session_state.active_claim_id or "None")

        st.markdown("---")

        # Claim Cards
        for cid, cstate in state.claims.items():
            claim = cstate.claim
            ctype = claim.claim_type.value if hasattr(claim.claim_type, "value") else str(claim.claim_type)
            is_active = (cid == st.session_state.active_claim_id)

            card_border = "#6366F1" if is_active else "#E2E8F0"
            active_badge = "🌟 **ACTIVE**" if is_active else ""

            with st.container():
                st.markdown(f"""
                <div class="claim-card" style="border-color: {card_border}; border-width: {'2px' if is_active else '1px'};">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-weight: 800; color: #4F46E5; font-size: 1.1rem;">[{cid}] {claim.subject or 'Proposition'}</span>
                        <span>
                            <span class="tag-pill" style="background-color: #E0E7FF; color: #3730A3;">{ctype.upper()}</span>
                            {active_badge}
                        </span>
                    </div>
                    <p style="font-size: 1rem; color: #1E293B; margin-bottom: 10px;"><b>Statement:</b> {claim.statement}</p>
                </div>
                """, unsafe_allow_html=True)

                ccol1, ccol2 = st.columns([4, 1])
                with ccol1:
                    tags_html = ""
                    if claim.benchmarks:
                        tags_html += "".join([f'<span class="tag-pill">📊 {b}</span>' for b in claim.benchmarks])
                    if claim.metrics:
                        tags_html += "".join([f'<span class="tag-pill">📈 {m}</span>' for m in claim.metrics])
                    if tags_html:
                        st.markdown(tags_html, unsafe_allow_html=True)
                with ccol2:
                    if st.button(f"Select [{cid}]", key=f"btn_select_{cid}", use_container_width=True):
                        st.session_state.active_claim_id = cid
                        state.active_claim_id = cid
                        st.rerun()

    else:
        st.info("Claims have not yet been extracted. Run the pipeline to view formalized propositions.")


# -----------------------------------------------------------------------------
# TAB 3: Evidence Explorer
# -----------------------------------------------------------------------------
with tab_evidence:
    st.header("🔍 Grounded Scientific Evidence")
    if state and state.global_evidence_store:
        st.write(f"Global Evidence Store contains **{len(state.global_evidence_store)} extracted evidence bundles**:")

        # Filter Options
        relationships = ["ALL", "SUPPORTS", "REPLICATES", "CONTRADICTS", "QUALIFIES"]
        selected_rel = st.selectbox("Filter Evidence by Relationship:", relationships)

        filtered_bundles = list(state.global_evidence_store.values())
        if selected_rel != "ALL":
            filtered_bundles = [b for b in filtered_bundles if getattr(b, "relationship", None) == selected_rel or getattr(getattr(b, "relationship", None), "value", "") == selected_rel]

        st.caption(f"Showing {len(filtered_bundles)} evidence item(s):")

        for bundle in filtered_bundles:
            rel = bundle.relationship.value if hasattr(bundle.relationship, "value") else str(bundle.relationship)
            rel_color = "#10B981" if rel in ["SUPPORTS", "REPLICATES"] else ("#EF4444" if rel == "CONTRADICTS" else "#F59E0B")

            with st.expander(f"[{bundle.id}] {bundle.source_title} ({rel})", expanded=False):
                st.markdown(f"""
                <span class="tag-pill" style="background-color: {rel_color}20; color: {rel_color}; font-weight: bold;">{rel}</span>
                <span class="tag-pill">📍 Location: {bundle.location or 'Paper'}</span>
                <span class="tag-pill">🎯 Linked Claim: {bundle.claim_id}</span>
                """, unsafe_allow_html=True)
                st.markdown(f"**Evidence Passage:**\n\n> {bundle.content}")
                if bundle.context:
                    st.caption(f"Context: {bundle.context}")
    else:
        st.info("No evidence units extracted yet. Run the pipeline to collect evidence bundles.")


# -----------------------------------------------------------------------------
# TAB 4: Dialectic Debate Arena
# -----------------------------------------------------------------------------
with tab_debate:
    st.header("⚔️ Adversarial Dialectic Debate Arena")
    active_id = st.session_state.active_claim_id
    if state and active_id and active_id in state.claims:
        cstate = state.claims[active_id]
        st.subheader(f"Debate on Claim [{active_id}]: \"{cstate.claim.statement}\"")

        col_sup, col_atk = st.columns(2)

        # Support Agent Column
        with col_sup:
            st.markdown("""
            <div class="debate-box support-box">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                    <span class="agent-pill agent-support">🛡️ SUPPORT AGENT (Proponent)</span>
                    <span style="font-weight: 700; color: #047857;">STANCE: FOR</span>
                </div>
            """, unsafe_allow_html=True)

            sup_arg = cstate.support_argument if hasattr(cstate, "support_argument") else cstate.get("support_argument")
            if sup_arg:
                st.markdown(f"**Argument Strength:** `{sup_arg.strength}`")
                st.markdown(f"**Conclusion:**\n> {sup_arg.conclusion}")
                st.markdown("**Core Affirmative Premises:**")
                for i, p in enumerate(sup_arg.premises, 1):
                    st.write(f"• {p}")
                if sup_arg.cited_evidence_ids:
                    st.markdown(f"**Cited Evidence Bundles:** `{'`, `'.join(sup_arg.cited_evidence_ids)}`")
            else:
                st.info("Support argument has not been constructed yet.")

            st.markdown("</div>", unsafe_allow_html=True)

        # Attack Agent Column
        with col_atk:
            st.markdown("""
            <div class="debate-box attack-box">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                    <span class="agent-pill agent-attack">⚔️ ATTACK AGENT (Adversary)</span>
                    <span style="font-weight: 700; color: #BE123C;">STANCE: AGAINST</span>
                </div>
            """, unsafe_allow_html=True)

            atk_arg = cstate.attack_argument if hasattr(cstate, "attack_argument") else cstate.get("attack_argument")
            if atk_arg:
                st.markdown(f"**Attack Strength:** `{atk_arg.strength}`")
                st.markdown(f"**Counter-Conclusion:**\n> {atk_arg.conclusion}")
                st.markdown("**Counter-Premises & Vulnerabilities:**")
                for i, p in enumerate(atk_arg.premises, 1):
                    st.write(f"• {p}")
                if atk_arg.identified_limitations:
                    st.markdown("**Identified Limitations:**")
                    for lim in atk_arg.identified_limitations:
                        st.caption(f"⚠️ {lim}")
                if atk_arg.cited_evidence_ids:
                    st.markdown("**External Grounded Citations:**")
                    for url in atk_arg.cited_evidence_ids:
                        st.markdown(f"- [{url}]({url})")
            else:
                st.info("Attack argument has not been constructed yet.")

            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Select a claim in the **Extracted Claims** tab to view the adversarial debate.")


# -----------------------------------------------------------------------------
# TAB 5: Final Verdict & Scientific Report
# -----------------------------------------------------------------------------
with tab_verdict:
    st.header("⚖️ Critic Verdict & Executive Report")
    active_id = st.session_state.active_claim_id
    if state and active_id and active_id in state.claims:
        cstate = state.claims[active_id]
        verdict_obj = cstate.verdict if hasattr(cstate, "verdict") else cstate.get("verdict")

        if verdict_obj:
            v_type = verdict_obj.verdict.value if hasattr(verdict_obj.verdict, "value") else str(verdict_obj.verdict)
            confidence = float(verdict_obj.confidence or 0.0)

            css_class = "verdict-inconclusive"
            badge_icon = "⚪"
            if v_type == "Supported":
                css_class = "verdict-supported"
                badge_icon = "🟢"
            elif v_type == "Partially Supported":
                css_class = "verdict-partially"
                badge_icon = "🟡"
            elif v_type == "Unsupported":
                css_class = "verdict-unsupported"
                badge_icon = "🔴"

            # Big Colorful Hero Verdict
            st.markdown(f"""
            <div class="verdict-card {css_class}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h2 style="margin: 0; font-size: 1.8rem;">{badge_icon} Final Verdict: {v_type}</h2>
                        <p style="margin: 4px 0 0 0; font-size: 1.05rem;">Evaluated on Claim [{active_id}]</p>
                    </div>
                    <div style="text-align: right;">
                        <h1 style="margin: 0; font-size: 2.2rem;">{int(confidence*100)}%</h1>
                        <span style="font-size: 0.85rem; font-weight: bold; text-transform: uppercase;">Confidence Score</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence Progress Bar
            st.progress(confidence)

            # Critic Diagnostic Checklist
            finding = verdict_obj.critic_finding if hasattr(verdict_obj, "critic_finding") else verdict_obj.get("critic_finding")
            if finding:
                st.subheader("🔍 Critic Agent Quality Audits")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    is_cv = getattr(finding, "citation_valid", False)
                    st.metric("Citation Grounding", "Verified ✅" if is_cv else "Unverified ❌")
                with c2:
                    is_rs = getattr(finding, "reasoning_sound", False)
                    st.metric("Reasoning Soundness", "Sound ✅" if is_rs else "Flawed ❌")
                with c3:
                    is_og = getattr(finding, "overgeneralization_detected", False)
                    st.metric("Overgeneralization", "Detected ⚠️" if is_og else "Clean ✅")
                with c4:
                    is_fc = getattr(finding, "fair_comparison", False)
                    st.metric("Comparative Parity", "Fair ✅" if is_fc else "Asymmetric ❌")

                if getattr(finding, "critique_notes", None):
                    st.info(f"**Critic Notes:** {finding.critique_notes}")

            # Scientific Synthesis Summary
            if verdict_obj.synthesis_summary:
                st.markdown("### 📝 Scientific Synthesis Summary")
                st.markdown(f"> {verdict_obj.synthesis_summary}")

        else:
            st.info("Verdict has not been synthesized for this claim yet.")

        # Full Executive Summary
        st.markdown("---")
        st.subheader("📑 Full Executive Scientific Summary")
        exec_sum = st.session_state.get("executive_summary")
        if exec_sum:
            st.markdown(exec_sum)

            # Export Buttons
            st.markdown("### 📥 Export Scientific Assessment")
            bcol1, bcol2 = st.columns(2)
            with bcol1:
                st.download_button(
                    label="📄 Download Report (Markdown)",
                    data=exec_sum,
                    file_name=f"mascv_report_{active_id}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with bcol2:
                report_json = {
                    "claim_id": active_id,
                    "paper_title": state.paper.metadata.title if state.paper and state.paper.metadata else "Unknown",
                    "claims": {cid: c.claim.model_dump() for cid, c in state.claims.items()},
                    "verdict": verdict_obj.model_dump() if verdict_obj and hasattr(verdict_obj, "model_dump") else None,
                    "executive_summary": exec_sum,
                }
                st.download_button(
                    label="💾 Download Raw Data (JSON)",
                    data=json.dumps(report_json, indent=2, default=str),
                    file_name=f"mascv_state_{active_id}.json",
                    mime="application/json",
                    use_container_width=True,
                )

    else:
        st.info("Run the pipeline or select a claim to view final scientific verdicts and reports.")

# -----------------------------------------------------------------------------
# Execution History Drawer
# -----------------------------------------------------------------------------
if st.session_state.execution_logs:
    with st.expander("📜 Live Multi-Agent Execution Log", expanded=False):
        for log in reversed(st.session_state.execution_logs):
            st.markdown(f"`{log['time']}` **[{log['agent']}]** {log['message']}")
