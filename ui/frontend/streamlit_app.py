"""Multi-Agent Scientific Claim Verification System (MASCV) - Research Dashboard."""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional

# Ensure src is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

import streamlit as st
import streamlit.components.v1 as components
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

/* Claim Cards */
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

/* Literature Search Card */
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

/* Markdown Tables */
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
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Auto-Scroll Functionality
# -----------------------------------------------------------------------------
def auto_scroll_to_bottom():
    """Injects JavaScript to smoothly scroll the active Streamlit viewport to the bottom."""
    ts = str(time.time())
    js_code = """
        <div id="scroll-anchor-__TS__"></div>
        <script>
        (function() {
            function findScrollContainer() {
                try {
                    const doc = window.parent.document;
                    if (!doc) return null;

                    // 1. In Streamlit 1.25+, stAppViewContainer is the primary scrolling container
                    const appView = doc.querySelector('[data-testid="stAppViewContainer"]');
                    if (appView && appView.scrollHeight > appView.clientHeight) {
                        return appView;
                    }

                    // 2. Candidate fallbacks
                    const candidates = [
                        appView,
                        doc.querySelector('section[data-testid="stMain"]'),
                        doc.querySelector('section.main'),
                        doc.querySelector('.main'),
                        doc.documentElement,
                        doc.body
                    ];

                    for (const el of candidates) {
                        if (el) {
                            const overflow = window.parent.getComputedStyle(el).overflowY;
                            if ((overflow === 'auto' || overflow === 'scroll' || overflow === 'overlay') && el.scrollHeight > el.clientHeight) {
                                return el;
                            }
                        }
                    }
                    return appView || doc.documentElement || doc.body;
                } catch (e) {
                    return null;
                }
            }

            function performScroll() {
                try {
                    // Priority 1: scrollIntoView on the iframe itself (cross-browser container-agnostic)
                    if (window.frameElement) {
                        window.frameElement.scrollIntoView({ behavior: 'smooth', block: 'end' });
                    }
                } catch (e) {}

                try {
                    // Priority 2: Direct container scroll
                    const container = findScrollContainer();
                    if (container) {
                        container.scrollTo({
                            top: container.scrollHeight + 10000,
                            behavior: 'smooth'
                        });
                    }

                    // Priority 3: Window-level scroll
                    if (window.parent && window.parent.scrollTo) {
                        window.parent.scrollTo({ top: 999999, behavior: 'smooth' });
                    }
                } catch (e) {}
            }

            // Run immediately and staged across React render cycles
            performScroll();
            setTimeout(performScroll, 50);
            setTimeout(performScroll, 200);
            setTimeout(performScroll, 500);

            // Attach persistent MutationObserver to stAppViewContainer
            try {
                const parentWin = window.parent;
                if (parentWin && !parentWin._mascv_scroll_attached) {
                    parentWin._mascv_scroll_attached = true;
                    const doc = parentWin.document;
                    const scrollTarget = doc.querySelector('[data-testid="stAppViewContainer"]') || doc.querySelector('section.main') || doc.body;
                    if (scrollTarget) {
                        let debounceTimer = null;
                        const observer = new MutationObserver(() => {
                            if (debounceTimer) clearTimeout(debounceTimer);
                            debounceTimer = setTimeout(performScroll, 80);
                        });
                        observer.observe(scrollTarget, { childList: true, subtree: true });
                    }
                }
            } catch (e) {}
        })();
        </script>
    """.replace("__TS__", ts)
    components.html(js_code, height=0, width=0)

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
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(field_name, default)
    return getattr(obj, field_name, default)


def format_evidence_item(b: Any) -> Dict[str, Any]:
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
# Multi-Agent Execution Handler (with Continuous Auto-Scroll)
# -----------------------------------------------------------------------------
if st.session_state.get("pipeline_running", False):
    auto_scroll_to_bottom()
    pdf_path = st.session_state.get("target_pdf")
    
    with st.status("🤖 Executing MASCV Multi-Agent Pipeline in Real Time...", expanded=True) as status_box:
        try:
            live_progress = st.progress(0.0)

            # 1. Paper Ingestion
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

            st.markdown("---")
            auto_scroll_to_bottom()

            # 2. Extract Claims
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
                st.markdown(f"""
                <div class="claim-card" style="border-left: 5px solid #6366F1; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 700; color: #4338CA;">[{cid}] {claim['subject'] or 'Scientific Proposition'}</span>
                        <span class="tag-pill" style="background-color: #E0E7FF; color: #3730A3;">{claim['claim_type'].upper()}</span>
                    </div>
                    <p style="margin: 6px 0 4px 0; color: #1E293B;"><b>Statement:</b> {claim['statement']}</p>
                </div>
                """, unsafe_allow_html=True)

            all_claim_ids = list(claims_dict.keys())
            total_claims = len(all_claim_ids)

            st.markdown("---")
            st.markdown(f"### 🔄 Investigating All Extracted Claims ({total_claims} Total)")
            auto_scroll_to_bottom()

            searcher = PaperSearchAgent()
            rag = EvidenceRAGAgent()
            support = SupportAgent()
            attack = AttackAgent()
            critic = CriticAgent()

            for claim_idx, active_id in enumerate(all_claim_ids, 1):
                st.markdown(f"## 🔬 Claim {claim_idx}/{total_claims}: `[{active_id}]`")
                st.session_state.active_claim_id = active_id
                if isinstance(state, dict):
                    state["active_claim_id"] = active_id
                elif hasattr(state, "active_claim_id"):
                    state.active_claim_id = active_id

                claims_dict = get_field(state, "claims", {})
                active_cstate = claims_dict.get(active_id, {}) if isinstance(claims_dict, dict) else getattr(claims_dict, active_id, {})
                active_claim = format_claim_item(get_field(active_cstate, "claim"))
                st.info(f"Target Proposition: **{active_claim['statement']}**")

                # 3. Paper Search
                st.markdown(f"#### 🌐 Phase 3: Literature Discovery (`PaperSearchAgent`)")
                with st.spinner(f"Executing Google Grounded Search for claim [{active_id}]..."):
                    state = searcher.execute(state)
                    claims_dict = get_field(state, "claims", {})
                    active_cstate = claims_dict.get(active_id, {}) if isinstance(claims_dict, dict) else getattr(claims_dict, active_id, {})
                    discovered_meta = get_field(active_cstate, "discovered_papers_metadata", [])
                    discovered_titles = get_field(active_cstate, "external_papers_found", [])
                    total_discovered = len(discovered_meta) if discovered_meta else len(discovered_titles)
                    add_log("PaperSearchAgent", f"Retrieved {total_discovered} external citations for [{active_id}].")

                st.success(f"🌐 Discovered **{total_discovered} relevant literature sources** for [{active_id}]")
                auto_scroll_to_bottom()

                # 4. Evidence RAG
                st.markdown(f"#### 📚 Phase 4: Grounded Evidence Extraction & Bundling (`EvidenceRAGAgent`)")
                with st.spinner(f"Extracting & verifying evidence passages for claim [{active_id}]..."):
                    state = rag.execute(state)
                    claims_dict = get_field(state, "claims", {})
                    active_cstate = claims_dict.get(active_id, {}) if isinstance(claims_dict, dict) else getattr(claims_dict, active_id, {})
                    bundle_ids = get_field(active_cstate, "evidence_bundle_ids", [])
                    add_log("EvidenceRAGAgent", f"Bundled {len(bundle_ids)} evidence passages for [{active_id}].")

                st.success(f"📚 Extracted & bundled **{len(bundle_ids)} evidence units** for [{active_id}]")
                auto_scroll_to_bottom()

                # 5. Support Agent
                st.markdown(f"#### 🛡️ Phase 5: Affirmative Case Construction (`SupportAgent`)")
                with st.spinner(f"Synthesizing affirmative argument with premises for [{active_id}]..."):
                    state = support.execute(state)
                    claims_dict = get_field(state, "claims", {})
                    active_cstate = claims_dict.get(active_id, {}) if isinstance(claims_dict, dict) else getattr(claims_dict, active_id, {})
                    raw_sup = get_field(active_cstate, "support_argument")
                    sup_arg = format_argument_item(raw_sup)
                    add_log("SupportAgent", f"Constructed affirmative argument for [{active_id}].")

                if sup_arg:
                    st.markdown(f"""
                    <div class="debate-box support-box">
                        <span class="agent-pill agent-support">🛡️ SUPPORT AGENT</span>
                        <span style="font-weight: 700; color: #047857;">STANCE: {sup_arg.get('stance', 'FOR')}</span>
                        <p style="font-weight: 600; color: #065F46; margin-top: 6px;">Conclusion: {sup_arg.get('conclusion', '')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                auto_scroll_to_bottom()

                # 6. Attack Agent
                st.markdown(f"#### ⚔️ Phase 6: Adversarial Attack & Boundary Testing (`AttackAgent`)")
                with st.spinner(f"Searching counter-evidence and probing vulnerabilities for [{active_id}]..."):
                    state = attack.execute(state)
                    claims_dict = get_field(state, "claims", {})
                    active_cstate = claims_dict.get(active_id, {}) if isinstance(claims_dict, dict) else getattr(claims_dict, active_id, {})
                    raw_atk = get_field(active_cstate, "attack_argument")
                    atk_arg = format_argument_item(raw_atk)
                    add_log("AttackAgent", f"Constructed adversarial counter-case for [{active_id}].")

                if atk_arg:
                    st.markdown(f"""
                    <div class="debate-box attack-box">
                        <span class="agent-pill agent-attack">⚔️ ATTACK AGENT</span>
                        <span style="font-weight: 700; color: #BE123C;">STANCE: {atk_arg.get('stance', 'AGAINST')}</span>
                        <p style="font-weight: 600; color: #9F1239; margin-top: 6px;">Counter-Conclusion: {atk_arg.get('conclusion', '')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                auto_scroll_to_bottom()

                # 7. Critic Agent
                st.markdown(f"#### ⚖️ Phase 7: Adjudicating Debate & Verdict for [{active_id}] (`CriticAgent`)")
                with st.spinner(f"Impartially evaluating debate & formulating verdict for [{active_id}]..."):
                    state = critic.execute(state)
                    claims_dict = get_field(state, "claims", {})
                    active_cstate = claims_dict.get(active_id, {}) if isinstance(claims_dict, dict) else getattr(claims_dict, active_id, {})
                    raw_verdict = get_field(active_cstate, "verdict")
                    verdict_dict = format_verdict_item(raw_verdict)
                    add_log("CriticAgent", f"Synthesized verdict for claim [{active_id}].")

                if verdict_dict:
                    v_type = verdict_dict.get("verdict", "Inconclusive")
                    conf = int((verdict_dict.get("confidence", 0.0) or 0.0) * 100)
                    st.markdown(f"""
                    <div style="background: #F8FAFC; border: 2px solid #6366F1; border-radius: 12px; padding: 14px 18px; margin-bottom: 10px;">
                        <h3 style="margin: 0; color: #4338CA;">⚖️ Final Verdict [{active_id}]: {v_type} ({conf}% Confidence)</h3>
                        <p style="margin: 6px 0 0 0; color: #334155;">{verdict_dict.get('synthesis_summary', '')}</p>
                    </div>
                    """, unsafe_allow_html=True)

                cur_prog = 0.25 + 0.65 * (claim_idx / total_claims)
                live_progress.progress(min(0.92, cur_prog))
                st.markdown("---")
                auto_scroll_to_bottom()

            # 8. Executive Summary
            st.markdown(f"### 🧭 Step 8/8: Overarching Scientific Executive Summary (`SupervisorAgent`)")
            with st.spinner("Synthesizing multi-agent executive assessment report across all claims..."):
                supervisor = SupervisorAgent()
                summary = supervisor.generate_executive_summary(state)
                st.session_state.executive_summary = summary
                st.session_state.pipeline_step = 8
                add_log("SupervisorAgent", f"Executive summary synthesized across all {total_claims} claims.")

            live_progress.progress(1.0)
            st.success("🧭 SupervisorAgent finalized report.")
            with st.expander("📄 Synthesized Executive Summary", expanded=True):
                st.markdown(summary)

            if all_claim_ids:
                st.session_state.active_claim_id = all_claim_ids[0]

            st.session_state.state = state
            st.session_state.pipeline_running = False
            status_box.update(label="🎉 Multi-Agent Pipeline Completed!", state="complete", expanded=True)
            st.balloons()
            auto_scroll_to_bottom()

        except Exception as exc:
            status_box.update(label=f"❌ Pipeline Failed: {exc}", state="error", expanded=True)
            st.session_state.pipeline_running = False
            st.error(f"Error during execution: {exc}")


# -----------------------------------------------------------------------------
# Main Dashboard Multi-View Tabs
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

# TAB 0: Live Activity Stream
with tab_stream:
    st.header("⚡ Real-Time Multi-Agent Activity Stream")
    claims_dict = get_field(state, "claims", {})
    if state and claims_dict:
        claim_keys = list(claims_dict.keys())
        default_index = claim_keys.index(st.session_state.active_claim_id) if st.session_state.active_claim_id in claim_keys else 0
        active_id = st.selectbox(
            "🎯 Select Verified Claim to Inspect:",
            claim_keys,
            index=default_index,
            format_func=lambda cid: f"[{cid}] {format_claim_item(get_field(claims_dict[cid], 'claim'))['statement'][:100]}...",
            key="stream_tab_claim_select",
        )
        st.session_state.active_claim_id = active_id

        s_cstate = claims_dict.get(active_id) if active_id else None
        s_verdict = format_verdict_item(get_field(s_cstate, "verdict"))
        v_str = s_verdict.get("verdict", "Pending") if s_verdict else "Pending"
        conf_str = f"{int(s_verdict.get('confidence', 0.0) * 100)}%" if s_verdict and s_verdict.get("confidence") else "N/A"

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total Claims", len(claims_dict))
        with m2:
            st.metric("Active Claim", active_id or "None")
        with m3:
            st.metric("Verdict", f"{v_str} ({conf_str})")
    else:
        st.info("👋 No multi-agent activity recorded yet. Run the pipeline from the sidebar to begin.")

# TAB 1: Paper Overview
with tab_paper:
    st.header("📄 Ingested Paper Overview")
    if state and state.paper:
        paper = state.paper
        st.subheader(paper.metadata.title or "Untitled Paper")
        if paper.metadata.authors:
            st.write(f"**Authors:** {', '.join(paper.metadata.authors)}")
        if paper.metadata.abstract:
            st.info(f"**Abstract:**\n\n{paper.metadata.abstract}")
    else:
        st.info("No publication loaded yet.")

# TAB 2: Extracted Claims
with tab_claims:
    st.header("🎯 Formalized Scientific Claims")
    if state and state.claims:
        for cid, cstate in state.claims.items():
            claim = format_claim_item(get_field(cstate, "claim"))
            st.markdown(f"**[{cid}]** {claim['statement']}")
    else:
        st.info("Claims have not yet been extracted.")

# TAB 3: Evidence Explorer
with tab_evidence:
    st.header("🔍 Grounded Scientific Evidence")
    if state and state.global_evidence_store:
        for b in state.global_evidence_store.values():
            item = format_evidence_item(b)
            st.markdown(f"- **[{item['relationship']}]** {item['source_title']}: {item['content'][:150]}...")
    else:
        st.info("No evidence units extracted yet.")

# TAB 4: Dialectic Debate Arena
with tab_debate:
    st.header("⚔️ Adversarial Dialectic Debate Arena")
    active_id = st.session_state.active_claim_id
    if state and active_id and active_id in state.claims:
        cstate = state.claims[active_id]
        sup_arg = format_argument_item(get_field(cstate, "support_argument"))
        atk_arg = format_argument_item(get_field(cstate, "attack_argument"))

        c_sup, c_atk = st.columns(2)
        with c_sup:
            st.markdown("### 🛡️ Proponent (Support)")
            if sup_arg:
                st.write(sup_arg["conclusion"])
        with c_atk:
            st.markdown("### ⚔️ Adversary (Attack)")
            if atk_arg:
                st.write(atk_arg["conclusion"])
    else:
        st.info("Select a claim to view debate.")

# TAB 5: Final Verdict & Scientific Report
with tab_verdict:
    st.header("⚖️ Critic Verdict & Executive Report")
    active_id = st.session_state.active_claim_id
    if state and active_id and active_id in state.claims:
        verdict_obj = format_verdict_item(get_field(state.claims[active_id], "verdict"))
        if verdict_obj:
            st.metric("Verdict", verdict_obj["verdict"], f"{int(verdict_obj['confidence']*100)}% Confidence")
            st.write(verdict_obj["synthesis_summary"])

        exec_sum = st.session_state.get("executive_summary")
        if exec_sum:
            st.markdown("### Executive Summary")
            st.markdown(exec_sum)
    else:
        st.info("Select a claim to view verdict.")

# Execution Log Drawer
if st.session_state.execution_logs:
    with st.expander("📜 Live Multi-Agent Execution Log", expanded=False):
        for log in reversed(st.session_state.execution_logs):
            st.markdown(f"`{log['time']}` **[{log['agent']}]** {log['message']}")