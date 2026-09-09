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
# Multi-Agent Execution Handler (with Live Progress)
# -----------------------------------------------------------------------------
if st.session_state.get("pipeline_running", False):
    pdf_path = st.session_state.get("target_pdf")
    
    with st.status("🤖 Executing MASCV Multi-Agent Pipeline...", expanded=True) as status_box:
        try:
            # 1. Ingest PDF
            st.write("📄 **[Step 1/8]** Ingesting & parsing target research paper with `PDFParser`...")
            parser = PDFParser()
            paper = parser.parse(pdf_path)
            state = InvestigationState(paper=paper)
            st.session_state.pipeline_step = 1
            add_log("PDFParser", f"Parsed '{paper.metadata.title}' into {len(paper.sections)} sections.")

            # 2. Extract Claims
            st.write("📋 **[Step 2/8]** Formalizing testable claims using `ClaimAnalystAgent` with Gemma 4...")
            analyst = ClaimAnalystAgent()
            state = analyst.execute(state)
            st.session_state.pipeline_step = 2
            add_log("ClaimAnalystAgent", f"Formulated {len(state.claims)} testable scientific claims.")

            # Select primary active claim (e.g. C1)
            active_id = state.active_claim_id or (list(state.claims.keys())[0] if state.claims else None)
            st.session_state.active_claim_id = active_id
            state.active_claim_id = active_id

            if active_id:
                # 3. Paper Search Agent
                st.write(f"🌐 **[Step 3/8]** Discovering external literature via `PaperSearchAgent` with Google Search Grounding...")
                searcher = PaperSearchAgent()
                state = searcher.execute(state)
                st.session_state.pipeline_step = 3
                discovered = state.claims[active_id].external_papers_found if hasattr(state.claims[active_id], "external_papers_found") else []
                add_log("PaperSearchAgent", f"Retrieved {len(discovered)} external citations/papers for claim [{active_id}].")

                # 4. Evidence RAG Agent
                st.write(f"📚 **[Step 4/8]** Extracting & bundling evidence units using `EvidenceRAGAgent`...")
                rag = EvidenceRAGAgent()
                state = rag.execute(state)
                st.session_state.pipeline_step = 4
                add_log("EvidenceRAGAgent", f"Bundled evidence passages in global store.")

                # 5. Support Agent
                st.write(f"🛡️ **[Step 5/8]** Formulating affirmative case with `SupportAgent`...")
                support = SupportAgent()
                state = support.execute(state)
                st.session_state.pipeline_step = 5
                add_log("SupportAgent", f"Constructed affirmative argument for [{active_id}].")

                # 6. Attack Agent
                st.write(f"⚔️ **[Step 6/8]** Searching counter-evidence & constructing attack case with `AttackAgent`...")
                attack = AttackAgent()
                state = attack.execute(state)
                st.session_state.pipeline_step = 6
                add_log("AttackAgent", f"Constructed adversarial counter-case for [{active_id}].")

                # 7. Critic Agent
                st.write(f"⚖️ **[Step 7/8]** Adjudicating debate & evaluating grounding with `CriticAgent`...")
                critic = CriticAgent()
                state = critic.execute(state)
                st.session_state.pipeline_step = 7
                add_log("CriticAgent", f"Synthesized verdict for claim [{active_id}].")

            # 8. Supervisor Executive Summary
            st.write("🧭 **[Step 8/8]** Synthesizing executive report with `SupervisorAgent`...")
            supervisor = SupervisorAgent()
            summary = supervisor.generate_executive_summary(state)
            st.session_state.executive_summary = summary
            st.session_state.pipeline_step = 8
            add_log("SupervisorAgent", "Executive summary synthesized.")

            st.session_state.state = state
            st.session_state.pipeline_running = False
            status_box.update(label="✅ Pipeline Execution Complete!", state="complete", expanded=False)
            st.rerun()

        except Exception as exc:
            status_box.update(label=f"❌ Pipeline Failed: {exc}", state="error", expanded=True)
            st.session_state.pipeline_running = False
            st.error(f"Error during execution: {exc}")


# -----------------------------------------------------------------------------
# Main Dashboard Multi-View Tabs
# -----------------------------------------------------------------------------
tab_paper, tab_claims, tab_evidence, tab_debate, tab_verdict = st.tabs([
    "📄 Paper Overview",
    "🎯 Extracted Claims",
    "🔍 Evidence Explorer",
    "⚔️ Dialectic Debate Arena",
    "⚖️ Verdict & Scientific Report",
])

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
                st.markdown(f"**{sec.heading or 'Section'}** ({len(sec.text.split())} words)")
                st.text(sec.text[:300] + ("..." if len(sec.text) > 300 else ""))
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
