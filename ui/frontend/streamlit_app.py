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
.verdict-card h1, .verdict-card h2, .verdict-card h3, .verdict-card p, .verdict-card span {
    color: inherit !important;
}
.verdict-supported {
    background-color: #ECFDF5 !important;
    border-color: #10B981 !important;
    color: #065F46 !important;
}
.verdict-partially {
    background-color: #FFFBEB !important;
    border-color: #F59E0B !important;
    color: #92400E !important;
}
.verdict-unsupported {
    background-color: #FEF2F2 !important;
    border-color: #EF4444 !important;
    color: #991B1B !important;
}
.verdict-inconclusive {
    background-color: #F8FAFC !important;
    border-color: #64748B !important;
    color: #334155 !important;
}

/* Side-by-Side Debate Cards */
.debate-box {
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}
.support-box {
    background: linear-gradient(180deg, #F0FDF4 0%, #FFFFFF 100%) !important;
    border: 2px solid #86EFAC !important;
    color: #065F46 !important;
}
.support-box p, .support-box span, .support-box div, .support-box b {
    color: #065F46 !important;
}
.attack-box {
    background: linear-gradient(180deg, #FFF1F2 0%, #FFFFFF 100%) !important;
    border: 2px solid #FDA4AF !important;
    color: #9F1239 !important;
}
.attack-box p, .attack-box span, .attack-box div, .attack-box b {
    color: #9F1239 !important;
}

/* Claim Cards (Explicit Contrast) */
.claim-card {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 12px;
    color: #0F172A !important;
    transition: all 0.2s ease;
}
.claim-card:hover {
    border-color: #6366F1 !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.1);
}
.claim-card p, .claim-card span, .claim-card div, .claim-card b {
    color: #0F172A !important;
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
    color: #334155 !important;
}

/* Literature Search Card (Explicit Contrast) */
.search-card {
    background: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-left: 4px solid #0284C7 !important;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
    color: #0F172A !important;
    transition: all 0.2s ease;
}
.search-card:hover {
    border-left-color: #0369A1 !important;
    box-shadow: 0 4px 12px rgba(2, 132, 199, 0.12);
}
.search-card p, .search-card span, .search-card div, .search-card b {
    color: #0F172A !important;
}

/* ------------------------------------------------------------------------- */
/* High-Contrast Markdown & Table Typography (Fixes White-on-White Anywhere) */
/* ------------------------------------------------------------------------- */

/* Global Markdown Tables (Executive Summary & Report Tables) */
div[data-testid="stMarkdownContainer"] table,
table {
    width: 100% !important;
    border-collapse: collapse !important;
    margin: 16px 0 !important;
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

div[data-testid="stMarkdownContainer"] th,
th {
    background-color: #EEF2FF !important;
    color: #1E1B4B !important;
    font-weight: 700 !important;
    padding: 10px 14px !important;
    border: 1px solid #CBD5E1 !important;
    text-align: left !important;
}

div[data-testid="stMarkdownContainer"] td,
td {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    padding: 10px 14px !important;
    border: 1px solid #E2E8F0 !important;
}

div[data-testid="stMarkdownContainer"] tr:nth-child(even) td,
tr:nth-child(even) td {
    background-color: #F8FAFC !important;
    color: #0F172A !important;
}

div[data-testid="stMarkdownContainer"] td strong,
div[data-testid="stMarkdownContainer"] td b,
div[data-testid="stMarkdownContainer"] th strong,
div[data-testid="stMarkdownContainer"] th b {
    color: #0F172A !important;
    font-weight: 700 !important;
}

/* Inline Code Elements inside Markdown & Tables */
div[data-testid="stMarkdownContainer"] code,
code {
    background-color: #F1F5F9 !important;
    color: #0F172A !important;
    font-weight: 600 !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    border: 1px solid #CBD5E1 !important;
}

/* Blockquotes (High Contrast) */
div[data-testid="stMarkdownContainer"] blockquote,
blockquote {
    background-color: #F8FAFC !important;
    border-left: 4px solid #6366F1 !important;
    color: #1E293B !important;
    padding: 12px 18px !important;
    border-radius: 0 8px 8px 0 !important;
    margin: 14px 0 !important;
}
div[data-testid="stMarkdownContainer"] blockquote p,
blockquote p {
    color: #1E293B !important;
    font-weight: 500 !important;
}

/* Expander Containers & Content (Guarantees Dark Text in Expanders) */
div[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 10px !important;
    margin-bottom: 12px !important;
}
div[data-testid="stExpander"] div[role="button"] {
    color: #0F172A !important;
    font-weight: 600 !important;
}
div[data-testid="stExpander"] div[role="button"] p,
div[data-testid="stExpander"] div[role="button"] span {
    color: #0F172A !important;
}
div[data-testid="stExpander"] div[data-testid="stMarkdownContainer"] p,
div[data-testid="stExpander"] div[data-testid="stMarkdownContainer"] li,
div[data-testid="stExpander"] div[data-testid="stMarkdownContainer"] span,
div[data-testid="stExpander"] div[data-testid="stMarkdownContainer"] div {
    color: #1E293B !important;
}
div[data-testid="stExpander"] div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stExpander"] div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stExpander"] div[data-testid="stMarkdownContainer"] h3,
div[data-testid="stExpander"] div[data-testid="stMarkdownContainer"] h4 {
    color: #312E81 !important;
}

/* Bordered Containers (e.g. Tab 5 Full Executive Summary) */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 10px !important;
    padding: 18px 24px !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] p,
div[data-testid="stVerticalBlockBorderWrapper"] li,
div[data-testid="stVerticalBlockBorderWrapper"] span,
div[data-testid="stVerticalBlockBorderWrapper"] div {
    color: #1E293B !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] h1,
div[data-testid="stVerticalBlockBorderWrapper"] h2,
div[data-testid="stVerticalBlockBorderWrapper"] h3,
div[data-testid="stVerticalBlockBorderWrapper"] h4 {
    color: #312E81 !important;
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


def get_field(obj: Any, field_name: str, default: Any = None) -> Any:
    """Safely extract field from Pydantic model, dataclass, or dictionary."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(field_name, default)
    return getattr(obj, field_name, default)


def format_evidence_item(b: Any) -> Dict[str, Any]:
    """Normalize evidence bundle whether it is a dict or EvidenceBundle Pydantic model."""
    if b is None:
        return {}
    b_id = get_field(b, "id", "E-?")
    source = get_field(b, "source_title", "Target Publication")
    location = get_field(b, "location", "Document")
    content = get_field(b, "content", "")
    context = get_field(b, "context", None)
    claim_id = get_field(b, "claim_id", "")
    raw_rel = get_field(b, "relationship", "SUPPORTS")
    rel_str = getattr(raw_rel, "value", str(raw_rel)) if raw_rel else "SUPPORTS"
    confidence = float(get_field(b, "confidence_score", 0.0) or 0.0)
    return {
        "id": b_id,
        "source_title": source,
        "location": location,
        "content": content,
        "context": context,
        "claim_id": claim_id,
        "relationship": rel_str,
        "confidence_score": confidence,
    }


def format_claim_item(claim_obj: Any) -> Dict[str, Any]:
    """Normalize claim whether it is a dict or Claim Pydantic model."""
    if claim_obj is None:
        return {}
    c_id = get_field(claim_obj, "id", "")
    statement = get_field(claim_obj, "statement", "")
    subject = get_field(claim_obj, "subject", "Scientific Proposition")
    raw_type = get_field(claim_obj, "claim_type", "performance")
    claim_type = getattr(raw_type, "value", str(raw_type)) if raw_type else "performance"
    benchmarks = get_field(claim_obj, "benchmarks", []) or []
    metrics = get_field(claim_obj, "metrics", []) or []
    return {
        "id": c_id,
        "statement": statement,
        "subject": subject,
        "claim_type": claim_type,
        "benchmarks": benchmarks,
        "metrics": metrics,
    }


def format_argument_item(arg_obj: Any) -> Optional[Dict[str, Any]]:
    """Normalize argument whether it is a dict or Argument Pydantic model."""
    if arg_obj is None:
        return None
    raw_premises = get_field(arg_obj, "premises", []) or []
    raw_limitations = get_field(arg_obj, "identified_limitations", []) or []
    raw_citations = get_field(arg_obj, "cited_evidence_ids", []) or []
    return {
        "stance": get_field(arg_obj, "stance", "FOR"),
        "strength": get_field(arg_obj, "strength", "Moderate"),
        "conclusion": get_field(arg_obj, "conclusion", ""),
        "premises": list(raw_premises),
        "cited_evidence_ids": list(raw_citations),
        "identified_limitations": list(raw_limitations),
    }


def format_verdict_item(verdict_obj: Any) -> Optional[Dict[str, Any]]:
    """Normalize verdict whether it is a dict or Verdict Pydantic model."""
    if verdict_obj is None:
        return None
    raw_v = get_field(verdict_obj, "verdict", "Inconclusive")
    v_type = getattr(raw_v, "value", str(raw_v)) if raw_v else "Inconclusive"
    confidence = float(get_field(verdict_obj, "confidence", 0.0) or 0.0)
    synthesis = get_field(verdict_obj, "synthesis_summary", "")
    finding = get_field(verdict_obj, "critic_finding", None)
    finding_dict = None
    if finding:
        finding_dict = {
            "citation_valid": bool(get_field(finding, "citation_valid", False)),
            "reasoning_sound": bool(get_field(finding, "reasoning_sound", False)),
            "overgeneralization_detected": bool(get_field(finding, "overgeneralization_detected", False)),
            "fair_comparison": bool(get_field(finding, "fair_comparison", False)),
            "critique_notes": str(get_field(finding, "critique_notes", "") or ""),
        }
    return {
        "verdict": v_type,
        "confidence": confidence,
        "synthesis_summary": synthesis,
        "critic_finding": finding_dict,
    }


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

            claims_dict = get_field(state, "claims", {})
            for cid, cstate in claims_dict.items():
                raw_c = get_field(cstate, "claim")
                claim = format_claim_item(raw_c)
                ctype = claim["claim_type"]
                subject = claim["subject"]
                statement = claim["statement"]
                benchmarks = claim["benchmarks"]
                metrics = claim["metrics"]
                st.markdown(f"""
                <div class="claim-card" style="border-left: 5px solid #6366F1; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 700; color: #4338CA;">[{cid}] {subject or 'Scientific Proposition'}</span>
                        <span class="tag-pill" style="background-color: #E0E7FF; color: #3730A3;">{ctype.upper()}</span>
                    </div>
                    <p style="margin: 6px 0 4px 0; color: #1E293B;"><b>Statement:</b> {statement}</p>
                </div>
                """, unsafe_allow_html=True)
                if benchmarks or metrics:
                    tags = " ".join([f"`📊 {b}`" for b in benchmarks] + [f"`📈 {m}`" for m in metrics])
                    st.caption(f"Grounding Targets: {tags}")

            # Get list of all formalized claims to evaluate
            claims_dict = get_field(state, "claims", {})
            all_claim_ids = list(claims_dict.keys())
            total_claims = len(all_claim_ids)

            st.markdown("---")
            st.markdown(f"### 🔄 Investigating All Extracted Claims ({total_claims} Total)")
            st.caption("Each claim undergoes independent Literature Grounded Search, Evidence Extraction, Support Case, Adversarial Attack, and Critic Peer Adjudication.")

            # Instantiate verification agents
            searcher = PaperSearchAgent()
            rag = EvidenceRAGAgent()
            support = SupportAgent()
            attack = AttackAgent()
            critic = CriticAgent()

            # Iterate through each claim sequentially
            for claim_idx, active_id in enumerate(all_claim_ids, 1):
                st.markdown(f"## 🔬 Claim {claim_idx}/{total_claims}: `[{active_id}]`")

                # Set active claim in state
                st.session_state.active_claim_id = active_id
                if isinstance(state, dict):
                    state["active_claim_id"] = active_id
                elif hasattr(state, "active_claim_id"):
                    state.active_claim_id = active_id

                claims_dict = get_field(state, "claims", {})
                active_cstate = claims_dict.get(active_id, {}) if isinstance(claims_dict, dict) else getattr(claims_dict, active_id, {})
                active_claim = format_claim_item(get_field(active_cstate, "claim"))
                st.info(f"Target Proposition: **{active_claim['statement']}**")

                # ---------------------------------------------------------
                # 3. Paper Search Agent
                # ---------------------------------------------------------
                st.markdown(f"#### 🌐 Phase 3: Literature Discovery (`PaperSearchAgent`)")
                with st.spinner(f"Executing Google Grounded Search for claim [{active_id}]..."):
                    state = searcher.execute(state)
                    claims_dict = get_field(state, "claims", {})
                    active_cstate = claims_dict.get(active_id, {}) if isinstance(claims_dict, dict) else getattr(claims_dict, active_id, {})
                    discovered_meta = get_field(active_cstate, "discovered_papers_metadata", [])
                    discovered_titles = get_field(active_cstate, "external_papers_found", [])
                    total_discovered = len(discovered_meta) if discovered_meta else len(discovered_titles)
                    add_log("PaperSearchAgent", f"Retrieved {total_discovered} external citations for [{active_id}].")

                st.success(f"🌐 Discovered **{total_discovered} relevant literature sources** for [{active_id}]:")

                if discovered_meta:
                    for p in discovered_meta:
                        p_title = get_field(p, "title", "Discovered Paper")
                        p_url = get_field(p, "url") or f"https://scholar.google.com/scholar?q={p_title.replace(' ', '+')}"
                        p_score_raw = get_field(p, "relevance_score", 0.85)
                        p_score = int((p_score_raw or 0.85) * 100)
                        p_rel = get_field(p, "relationship", "RELEVANT")
                        p_authors = get_field(p, "authors", [])
                        p_year = get_field(p, "year", "N/A")
                        p_venue = get_field(p, "venue", "Repository")
                        p_findings = get_field(p, "relevance_rationale") or get_field(p, "key_findings")
                        st.markdown(f"""
                        <div class="search-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <a href="{p_url}" target="_blank" style="font-weight: 700; color: #0284C7; text-decoration: none;">📄 {p_title}</a>
                                <span class="tag-pill" style="background: #E0F2FE; color: #0369A1;">{p_rel} • {p_score}% RELEVANCE</span>
                            </div>
                            <div style="font-size: 0.85rem; color: #64748B; margin-top: 4px;">
                                Authors: {', '.join(p_authors[:3]) if p_authors else 'Academic Authors'} ({p_year}) • Venue: {p_venue}
                            </div>
                            {f'<div style="font-size: 0.85rem; color: #334155; margin-top: 4px; font-style: italic;"><b>Findings:</b> {p_findings}</div>' if p_findings else ''}
                        </div>
                        """, unsafe_allow_html=True)
                elif discovered_titles:
                    for t in discovered_titles:
                        st.markdown(f"- 📄 **{t}**")

                # ---------------------------------------------------------
                # 4. Evidence RAG Agent
                # ---------------------------------------------------------
                st.markdown(f"#### 📚 Phase 4: Grounded Evidence Extraction & Bundling (`EvidenceRAGAgent`)")
                with st.spinner(f"Extracting & verifying evidence passages for claim [{active_id}]..."):
                    state = rag.execute(state)
                    ev_store = get_field(state, "global_evidence_store", {})
                    claims_dict = get_field(state, "claims", {})
                    active_cstate = claims_dict.get(active_id, {}) if isinstance(claims_dict, dict) else getattr(claims_dict, active_id, {})
                    bundle_ids = get_field(active_cstate, "evidence_bundle_ids", [])
                    add_log("EvidenceRAGAgent", f"Bundled {len(bundle_ids)} evidence passages for [{active_id}].")

                st.success(f"📚 Extracted & bundled **{len(bundle_ids)} evidence units** for [{active_id}]:")

                with st.expander(f"🔍 Inspect Grounded Evidence Bundles for [{active_id}] ({len(bundle_ids)} total)", expanded=False):
                    for bid in bundle_ids[:4]:
                        raw_b = ev_store.get(bid) if isinstance(ev_store, dict) else None
                        if raw_b:
                            b = format_evidence_item(raw_b)
                            b_rel = b["relationship"]
                            b_color = "#10B981" if b_rel in ["SUPPORTS", "REPLICATES"] else ("#EF4444" if b_rel == "CONTRADICTS" else "#F59E0B")
                            st.markdown(f"""
                            <div style="border-left: 3px solid {b_color}; padding-left: 10px; margin-bottom: 10px;">
                                <span class="tag-pill" style="background-color: {b_color}20; color: {b_color}; font-weight: bold;">{b_rel}</span>
                                <span style="font-size: 0.85rem; color: #64748B;">Source: <b>{b['source_title']}</b> ({b['location'] or 'Document'})</span>
                                <p style="margin: 4px 0 0 0; font-size: 0.9rem; color: #1E293B;">"{b['content'][:240]}..."</p>
                            </div>
                            """, unsafe_allow_html=True)

                # ---------------------------------------------------------
                # 5. Support Agent
                # ---------------------------------------------------------
                st.markdown(f"#### 🛡️ Phase 5: Affirmative Case Construction (`SupportAgent`)")
                with st.spinner(f"Synthesizing affirmative argument with premises for [{active_id}]..."):
                    state = support.execute(state)
                    claims_dict = get_field(state, "claims", {})
                    active_cstate = claims_dict.get(active_id, {}) if isinstance(claims_dict, dict) else getattr(claims_dict, active_id, {})
                    raw_sup = get_field(active_cstate, "support_argument")
                    sup_arg = format_argument_item(raw_sup)
                    add_log("SupportAgent", f"Constructed affirmative argument for [{active_id}].")

                if sup_arg:
                    st_strength = str(sup_arg.get("strength") or "MODERATE").upper()
                    st.markdown(f"""
                    <div class="debate-box support-box">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span class="agent-pill agent-support">🛡️ SUPPORT AGENT</span>
                            <span style="font-weight: 700; color: #047857;">STANCE: {sup_arg.get('stance', 'FOR')} • STRENGTH: {st_strength}</span>
                        </div>
                        <p style="font-weight: 600; color: #065F46; margin-bottom: 6px;">Conclusion: {sup_arg.get('conclusion', '')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    sup_premises = sup_arg.get("premises", [])
                    sup_citations = sup_arg.get("cited_evidence_ids", [])
                    if sup_premises:
                        with st.expander(f"View Support Affirmative Premises [{active_id}]", expanded=False):
                            for p in sup_premises:
                                st.write(f"• {p}")
                            if sup_citations:
                                st.caption(f"Cited Evidence IDs: {', '.join(sup_citations)}")

                # ---------------------------------------------------------
                # 6. Attack Agent
                # ---------------------------------------------------------
                st.markdown(f"#### ⚔️ Phase 6: Adversarial Attack & Boundary Testing (`AttackAgent`)")
                with st.spinner(f"Searching counter-evidence and probing vulnerabilities for [{active_id}]..."):
                    state = attack.execute(state)
                    claims_dict = get_field(state, "claims", {})
                    active_cstate = claims_dict.get(active_id, {}) if isinstance(claims_dict, dict) else getattr(claims_dict, active_id, {})
                    raw_atk = get_field(active_cstate, "attack_argument")
                    atk_arg = format_argument_item(raw_atk)
                    add_log("AttackAgent", f"Constructed adversarial counter-case for [{active_id}].")

                if atk_arg:
                    atk_strength = str(atk_arg.get("strength") or "MODERATE").upper()
                    st.markdown(f"""
                    <div class="debate-box attack-box">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span class="agent-pill agent-attack">⚔️ ATTACK AGENT</span>
                            <span style="font-weight: 700; color: #BE123C;">STANCE: {atk_arg.get('stance', 'AGAINST')} • STRENGTH: {atk_strength}</span>
                        </div>
                        <p style="font-weight: 600; color: #9F1239; margin-bottom: 6px;">Counter-Conclusion: {atk_arg.get('conclusion', '')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    atk_premises = atk_arg.get("premises", [])
                    atk_limitations = atk_arg.get("identified_limitations", [])
                    atk_citations = atk_arg.get("cited_evidence_ids", [])
                    if atk_premises or atk_limitations:
                        with st.expander(f"View Attack Counter-Premises & Vulnerabilities [{active_id}]", expanded=False):
                            for p in atk_premises:
                                st.write(f"• {p}")
                            if atk_limitations:
                                st.write("**Identified Limitations:**")
                                for lim in atk_limitations:
                                    st.caption(f"⚠️ {lim}")
                            if atk_citations:
                                st.write("**External Counter-Evidence Links:**")
                                for link in atk_citations:
                                    st.markdown(f"- [{link}]({link})")

                # ---------------------------------------------------------
                # 7. Critic Agent
                # ---------------------------------------------------------
                st.markdown(f"#### ⚖️ Phase 7: Adjudicating Debate & Verdict for [{active_id}] (`CriticAgent`)")
                with st.spinner(f"Impartially evaluating debate, verifying citations & formulating verdict for [{active_id}]..."):
                    state = critic.execute(state)
                    claims_dict = get_field(state, "claims", {})
                    active_cstate = claims_dict.get(active_id, {}) if isinstance(claims_dict, dict) else getattr(claims_dict, active_id, {})
                    raw_verdict = get_field(active_cstate, "verdict")
                    verdict_dict = format_verdict_item(raw_verdict)
                    add_log("CriticAgent", f"Synthesized verdict for claim [{active_id}].")

                if verdict_dict:
                    v_type = verdict_dict.get("verdict", "Inconclusive")
                    conf = int((verdict_dict.get("confidence", 0.0) or 0.0) * 100)
                    synthesis = verdict_dict.get("synthesis_summary", "")
                    finding = verdict_dict.get("critic_finding")

                    st.markdown(f"""
                    <div style="background: #F8FAFC; border: 2px solid #6366F1; border-radius: 12px; padding: 14px 18px; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h3 style="margin: 0; color: #4338CA;">⚖️ Final Verdict [{active_id}]: {v_type}</h3>
                            <span style="font-size: 1.2rem; font-weight: 800; color: #4338CA;">{conf}% Confidence</span>
                        </div>
                        <p style="margin: 8px 0 0 0; color: #334155; font-size: 0.95rem;"><b>Scientific Synthesis:</b> {synthesis}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    if finding:
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            st.metric("Citation Grounding", "Verified ✅" if finding.get("citation_valid") else "Unverified ❌")
                        with c2:
                            st.metric("Reasoning Soundness", "Sound ✅" if finding.get("reasoning_sound") else "Flawed ❌")
                        with c3:
                            st.metric("Overgeneralization", "Clean ✅" if not finding.get("overgeneralization_detected") else "Detected ⚠️")
                        with c4:
                            st.metric("Comparative Parity", "Fair ✅" if finding.get("fair_comparison") else "Asymmetric ❌")

                # Step progress within multi-claim execution
                cur_prog = 0.25 + 0.65 * (claim_idx / total_claims)
                live_progress.progress(min(0.92, cur_prog))
                st.markdown("---")

            # ---------------------------------------------------------
            # 8. Supervisor Executive Summary (AT THE VERY END AFTER ALL CLAIMS ARE REVIEWED)
            # ---------------------------------------------------------
            st.markdown(f"### 🧭 Step 8/8: Overarching Scientific Executive Summary (`SupervisorAgent`)")
            st.info(f"Synthesizing meta-analysis across all **{total_claims} verified propositions**...")
            with st.spinner("Synthesizing multi-agent executive assessment report across all claims..."):
                supervisor = SupervisorAgent()
                summary = supervisor.generate_executive_summary(state)
                st.session_state.executive_summary = summary
                st.session_state.pipeline_step = 8
                add_log("SupervisorAgent", f"Executive summary synthesized across all {total_claims} claims.")

            live_progress.progress(1.0)
            st.success(f"🧭 **SupervisorAgent** synthesized final assessment report across all {total_claims} claims.")
            with st.expander("📄 Preview Synthesized Executive Summary", expanded=True):
                with st.container(border=True):
                    st.markdown(summary)

            # Set active claim back to first claim for clean tab exploration
            if all_claim_ids:
                st.session_state.active_claim_id = all_claim_ids[0]
                if isinstance(state, dict):
                    state["active_claim_id"] = all_claim_ids[0]
                elif hasattr(state, "active_claim_id"):
                    state.active_claim_id = all_claim_ids[0]

            # Completion & state persistence
            st.session_state.state = state
            st.session_state.pipeline_running = False
            st.session_state.pipeline_completed = True
            status_box.update(label="🎉 Multi-Agent Pipeline Execution Succeeded (All Claims Reviewed)!", state="complete", expanded=True)
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
    claims_dict = get_field(state, "claims", {})
    if state and claims_dict:
        st.write("Complete chronological trace of all 8 multi-agent verification phases:")

        # Interactive Claim Switcher for Multi-Claim Trace
        claim_keys = list(claims_dict.keys())
        default_index = claim_keys.index(st.session_state.active_claim_id) if st.session_state.active_claim_id in claim_keys else 0
        active_id = st.selectbox(
            "🎯 Select Verified Claim to Inspect Verification Trace & Dialectic Debate:",
            claim_keys,
            index=default_index,
            format_func=lambda cid: f"[{cid}] {format_claim_item(get_field(claims_dict[cid], 'claim'))['statement'][:100]}...",
            key="stream_tab_claim_select",
        )
        st.session_state.active_claim_id = active_id

        # Summary Metric Bar
        s_cstate = claims_dict.get(active_id) if active_id else None
        s_verdict = format_verdict_item(get_field(s_cstate, "verdict"))
        v_str = s_verdict.get("verdict", "Pending") if s_verdict else "Pending"
        conf_str = f"{int(s_verdict.get('confidence', 0.0) * 100)}%" if s_verdict and s_verdict.get("confidence") else "N/A"

        ev_store = get_field(state, "global_evidence_store", {})
        ev_count = len(ev_store) if isinstance(ev_store, (dict, list)) else 0

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Total Claims", len(claims_dict))
        with m2:
            st.metric("Active Claim", active_id or "None")
        with m3:
            papers_found_count = len(get_field(s_cstate, "discovered_papers_metadata", [])) or len(get_field(s_cstate, "external_papers_found", [])) if s_cstate else 0
            st.metric("Discovered Sources", papers_found_count)
        with m4:
            st.metric("Evidence Bundles", ev_count)
        with m5:
            st.metric("Active Verdict", f"{v_str} ({conf_str})")

        st.markdown("---")

        # 1. Ingested Paper Summary
        with st.expander("📄 Phase 1: Ingested Paper & Structure (`PDFParser`)", expanded=True):
            paper_obj = get_field(state, "paper")
            if paper_obj:
                p_meta = get_field(paper_obj, "metadata")
                p_title = get_field(p_meta, "title", "Untitled Paper")
                p_authors = get_field(p_meta, "authors", [])
                st.markdown(f"**Publication Title:** {p_title or 'Untitled Paper'}")
                if p_authors:
                    st.caption(f"Authors: {', '.join(p_authors)}")
                sections = get_field(paper_obj, "sections", [])
                raw_text = get_field(paper_obj, "raw_text", "")
                st.write(f"Parsed **{len(sections)} sections** across **{len(raw_text.split()):,} words**.")

        # 2. Extracted Claims
        with st.expander(f"🎯 Phase 2: Formalized Scientific Claims ({len(claims_dict)} extracted via `ClaimAnalystAgent`)", expanded=True):
            for cid, cstate in claims_dict.items():
                c = format_claim_item(get_field(cstate, "claim"))
                ctype = c["claim_type"]
                subject = c["subject"]
                statement = c["statement"]
                is_act = (cid == active_id)
                card_bg = "#EEF2FF" if is_act else "#F8FAFC"
                st.markdown(f"""
                <div style="background: {card_bg}; border: 1px solid #CBD5E1; border-left: 4px solid {'#4F46E5' if is_act else '#94A3B8'}; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between;">
                        <b>[{cid}] {subject or 'Scientific Proposition'}</b>
                        <span class="tag-pill" style="background: #E0E7FF; color: #3730A3;">{ctype.upper()}</span>
                    </div>
                    <p style="margin: 4px 0; color: #1E293B;"><b>Statement:</b> {statement}</p>
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
            for raw_b in list(state.global_evidence_store.values())[:6]:
                b = format_evidence_item(raw_b)
                b_rel = b["relationship"]
                b_color = "#10B981" if b_rel in ["SUPPORTS", "REPLICATES"] else ("#EF4444" if b_rel == "CONTRADICTS" else "#F59E0B")
                st.markdown(f"""
                <div style="border-left: 3px solid {b_color}; padding-left: 10px; margin-bottom: 8px;">
                    <span class="tag-pill" style="background-color: {b_color}20; color: {b_color}; font-weight: bold;">{b_rel}</span>
                    <span style="font-size: 0.85rem; color: #64748B;">Source: <b>{b['source_title']}</b> ({b['location'] or 'Document'})</span>
                    <p style="margin: 2px 0 0 0; font-size: 0.88rem; color: #1E293B;">"{b['content'][:220]}..."</p>
                </div>
                """, unsafe_allow_html=True)

        # 5 & 6. Dialectic Debate
        if s_cstate:
            with st.expander("⚔️ Phases 5 & 6: Dialectic Debate Arena (`SupportAgent` vs `AttackAgent`)", expanded=True):
                dcol1, dcol2 = st.columns(2)
                with dcol1:
                    raw_sup = s_cstate.get("support_argument") if isinstance(s_cstate, dict) else getattr(s_cstate, "support_argument", None)
                    sup = format_argument_item(raw_sup)
                    if sup:
                        st.markdown(f"""
                        <div class="debate-box support-box">
                            <span class="agent-pill agent-support">🛡️ SUPPORT AGENT</span>
                            <span style="font-weight: 700; color: #047857;">STANCE: FOR ({sup['strength']})</span>
                            <p style="margin-top: 6px; font-weight: 600;">{sup['conclusion']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        for p in sup["premises"]:
                            st.caption(f"• {p}")
                with dcol2:
                    raw_atk = s_cstate.get("attack_argument") if isinstance(s_cstate, dict) else getattr(s_cstate, "attack_argument", None)
                    atk = format_argument_item(raw_atk)
                    if atk:
                        st.markdown(f"""
                        <div class="debate-box attack-box">
                            <span class="agent-pill agent-attack">⚔️ ATTACK AGENT</span>
                            <span style="font-weight: 700; color: #BE123C;">STANCE: AGAINST ({atk['strength']})</span>
                            <p style="margin-top: 6px; font-weight: 600;">{atk['conclusion']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        for p in atk["premises"]:
                            st.caption(f"• {p}")

        # 7. Critic Verdict
        v_formatted = format_verdict_item(s_verdict)
        if v_formatted:
            with st.expander("⚖️ Phase 7: Critic Agent Adjudication & 4-Point Peer Audit", expanded=True):
                v_type = v_formatted["verdict"]
                v_conf = f"{int(v_formatted['confidence'] * 100)}%"
                st.markdown(f"**Final Synthesized Verdict:** `{v_type}` ({v_conf} Confidence)")
                if v_formatted["synthesis_summary"]:
                    st.info(f"**Scientific Synthesis:** {v_formatted['synthesis_summary']}")
                f_obj = v_formatted.get("critic_finding")
                if f_obj:
                    qc1, qc2, qc3, qc4 = st.columns(4)
                    with qc1:
                        st.metric("Grounding", "Verified ✅" if f_obj.get("citation_valid") else "Unverified ❌")
                    with qc2:
                        st.metric("Soundness", "Sound ✅" if f_obj.get("reasoning_sound") else "Flawed ❌")
                    with qc3:
                        st.metric("Overgeneralization", "Clean ✅" if not f_obj.get("overgeneralization_detected") else "Detected ⚠️")
                    with qc4:
                        st.metric("Parity", "Fair ✅" if f_obj.get("fair_comparison") else "Asymmetric ❌")

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
            raw_claim = c.get("claim") if isinstance(c, dict) else getattr(c, "claim", None)
            c_dict = format_claim_item(raw_claim)
            ctype = c_dict["claim_type"]
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
            raw_claim = cstate.get("claim") if isinstance(cstate, dict) else getattr(cstate, "claim", None)
            claim = format_claim_item(raw_claim)
            ctype = claim["claim_type"]
            is_active = (cid == st.session_state.active_claim_id)

            card_border = "#6366F1" if is_active else "#E2E8F0"
            active_badge = "🌟 **ACTIVE**" if is_active else ""

            with st.container():
                st.markdown(f"""
                <div class="claim-card" style="border-color: {card_border}; border-width: {'2px' if is_active else '1px'};">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-weight: 800; color: #4F46E5; font-size: 1.1rem;">[{cid}] {claim['subject'] or 'Proposition'}</span>
                        <span>
                            <span class="tag-pill" style="background-color: #E0E7FF; color: #3730A3;">{ctype.upper()}</span>
                            {active_badge}
                        </span>
                    </div>
                    <p style="font-size: 1rem; color: #1E293B; margin-bottom: 10px;"><b>Statement:</b> {claim['statement']}</p>
                </div>
                """, unsafe_allow_html=True)

                ccol1, ccol2 = st.columns([4, 1])
                with ccol1:
                    tags_html = ""
                    if claim["benchmarks"]:
                        tags_html += "".join([f'<span class="tag-pill">📊 {b}</span>' for b in claim["benchmarks"]])
                    if claim["metrics"]:
                        tags_html += "".join([f'<span class="tag-pill">📈 {m}</span>' for m in claim["metrics"]])
                    if tags_html:
                        st.markdown(tags_html, unsafe_allow_html=True)
                with ccol2:
                    if st.button(f"Select [{cid}]", key=f"btn_select_{cid}", use_container_width=True):
                        st.session_state.active_claim_id = cid
                        if hasattr(state, "active_claim_id"):
                            state.active_claim_id = cid
                        elif isinstance(state, dict):
                            state["active_claim_id"] = cid
                        st.rerun()

    else:
        st.info("Claims have not yet been extracted. Run the pipeline to view formalized propositions.")


# -----------------------------------------------------------------------------
# TAB 3: Evidence Explorer
# -----------------------------------------------------------------------------
with tab_evidence:
    st.header("🔍 Grounded Scientific Evidence")
    if state and state.global_evidence_store:
        raw_bundles = [format_evidence_item(b) for b in state.global_evidence_store.values()]
        st.write(f"Global Evidence Store contains **{len(raw_bundles)} extracted evidence bundles**:")

        # Filter Options
        relationships = ["ALL", "SUPPORTS", "REPLICATES", "CONTRADICTS", "QUALIFIES"]
        selected_rel = st.selectbox("Filter Evidence by Relationship:", relationships)

        filtered_bundles = raw_bundles
        if selected_rel != "ALL":
            filtered_bundles = [b for b in raw_bundles if b["relationship"] == selected_rel]

        st.caption(f"Showing {len(filtered_bundles)} evidence item(s):")

        for bundle in filtered_bundles:
            rel = bundle["relationship"]
            rel_color = "#10B981" if rel in ["SUPPORTS", "REPLICATES"] else ("#EF4444" if rel == "CONTRADICTS" else "#F59E0B")

            with st.expander(f"[{bundle['id']}] {bundle['source_title']} ({rel})", expanded=False):
                st.markdown(f"""
                <span class="tag-pill" style="background-color: {rel_color}20; color: {rel_color}; font-weight: bold;">{rel}</span>
                <span class="tag-pill">📍 Location: {bundle['location'] or 'Paper'}</span>
                <span class="tag-pill">🎯 Linked Claim: {bundle['claim_id']}</span>
                """, unsafe_allow_html=True)
                st.markdown(f"**Evidence Passage:**\n\n> {bundle['content']}")
                if bundle["context"]:
                    st.caption(f"Context: {bundle['context']}")
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
        claim_obj = cstate.get("claim") if isinstance(cstate, dict) else getattr(cstate, "claim", None)
        claim_stmt = get_field(claim_obj, "statement", "")
        st.subheader(f"Debate on Claim [{active_id}]: \"{claim_stmt}\"")

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

            raw_sup = cstate.get("support_argument") if isinstance(cstate, dict) else getattr(cstate, "support_argument", None)
            sup_arg = format_argument_item(raw_sup)
            if sup_arg:
                st.markdown(f"**Argument Strength:** `{sup_arg['strength']}`")
                st.markdown(f"**Conclusion:**\n> {sup_arg['conclusion']}")
                st.markdown("**Core Affirmative Premises:**")
                for i, p in enumerate(sup_arg["premises"], 1):
                    st.write(f"• {p}")
                if sup_arg["cited_evidence_ids"]:
                    st.markdown(f"**Cited Evidence Bundles:** `{'`, `'.join(sup_arg['cited_evidence_ids'])}`")
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

            raw_atk = cstate.get("attack_argument") if isinstance(cstate, dict) else getattr(cstate, "attack_argument", None)
            atk_arg = format_argument_item(raw_atk)
            if atk_arg:
                st.markdown(f"**Attack Strength:** `{atk_arg['strength']}`")
                st.markdown(f"**Counter-Conclusion:**\n> {atk_arg['conclusion']}")
                st.markdown("**Counter-Premises & Vulnerabilities:**")
                for i, p in enumerate(atk_arg["premises"], 1):
                    st.write(f"• {p}")
                if atk_arg["identified_limitations"]:
                    st.markdown("**Identified Limitations:**")
                    for lim in atk_arg["identified_limitations"]:
                        st.caption(f"⚠️ {lim}")
                if atk_arg["cited_evidence_ids"]:
                    st.markdown("**External Grounded Citations:**")
                    for url in atk_arg["cited_evidence_ids"]:
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
        raw_v = cstate.get("verdict") if isinstance(cstate, dict) else getattr(cstate, "verdict", None)
        verdict_obj = format_verdict_item(raw_v)

        if verdict_obj:
            v_type = verdict_obj["verdict"]
            confidence = float(verdict_obj["confidence"] or 0.0)

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
            finding = verdict_obj.get("critic_finding")
            if finding:
                st.subheader("🔍 Critic Agent Quality Audits")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    is_cv = finding.get("citation_valid", False)
                    st.metric("Citation Grounding", "Verified ✅" if is_cv else "Unverified ❌")
                with c2:
                    is_rs = finding.get("reasoning_sound", False)
                    st.metric("Reasoning Soundness", "Sound ✅" if is_rs else "Flawed ❌")
                with c3:
                    is_og = finding.get("overgeneralization_detected", False)
                    st.metric("Overgeneralization", "Detected ⚠️" if is_og else "Clean ✅")
                with c4:
                    is_fc = finding.get("fair_comparison", False)
                    st.metric("Comparative Parity", "Fair ✅" if is_fc else "Asymmetric ❌")

                if finding.get("critique_notes"):
                    st.info(f"**Critic Notes:** {finding['critique_notes']}")

            # Scientific Synthesis Summary
            if verdict_obj["synthesis_summary"]:
                st.markdown("### 📝 Scientific Synthesis Summary")
                st.markdown(f"> {verdict_obj['synthesis_summary']}")

        else:
            st.info("Verdict has not been synthesized for this claim yet.")

        # Full Executive Summary
        st.markdown("---")
        st.subheader("📑 Full Executive Scientific Summary")
        exec_sum = st.session_state.get("executive_summary")
        if exec_sum:
            with st.container(border=True):
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
                paper_title = "Unknown"
                if state.paper and getattr(state.paper, "metadata", None):
                    paper_title = getattr(state.paper.metadata, "title", "Unknown")
                claims_export = {}
                for cid, c in state.claims.items():
                    c_claim = c.get("claim") if isinstance(c, dict) else getattr(c, "claim", None)
                    claims_export[cid] = c_claim.model_dump() if hasattr(c_claim, "model_dump") else c_claim
                report_json = {
                    "claim_id": active_id,
                    "paper_title": paper_title,
                    "claims": claims_export,
                    "verdict": verdict_obj,
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
