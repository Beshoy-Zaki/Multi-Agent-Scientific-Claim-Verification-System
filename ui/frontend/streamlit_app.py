"""Streamlit user interface for MASCV dashboard."""

import streamlit as st


def main():
    st.set_page_config(
        page_title="MASCV - Multi-Agent Scientific Claim Verification",
        page_icon="🔬",
        layout="wide",
    )

    st.title("🔬 Multi-Agent Scientific Claim Verification System")
    st.markdown(
        "An Evidence-Grounded Multi-Agent System for Adversarial Analysis of Scientific Claims."
    )

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select View",
        ["1. Paper Overview", "2. Claims Dashboard", "3. Evidence View", "4. Debate View", "5. Final Verdict & Report"],
    )

    if page == "1. Paper Overview":
        st.header("📄 Paper Overview & Ingestion")
        uploaded_file = st.file_uploader("Upload Target Paper (PDF)", type=["pdf"])
        if uploaded_file:
            st.success(f"Loaded: {uploaded_file.name}")
            st.info("Agentic pipeline ready to extract propositions.")

    elif page == "2. Claims Dashboard":
        st.header("📋 Extracted Scientific Claims")
        st.write("Propositions extracted by the Claim Analyst agent.")

    elif page == "3. Evidence View":
        st.header("🔍 Retrieved Literature & Evidence Bundles")
        st.write("Supporting and contradictory evidence gathered from academic sources.")

    elif page == "4. Debate View":
        st.header("⚖️ Adversarial Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🛡️ Support Agent")
            st.write("Affirmative case and independent replications.")
        with col2:
            st.subheader("⚔️ Attack Agent")
            st.write("Counter-evidence, limitations, and methodological weaknesses.")

    elif page == "5. Final Verdict & Report":
        st.header("⚖️ Critic Assessment & Final Scientific Report")
        st.write("Citation validation, overgeneralization check, and verdict.")


if __name__ == "__main__":
    main()
