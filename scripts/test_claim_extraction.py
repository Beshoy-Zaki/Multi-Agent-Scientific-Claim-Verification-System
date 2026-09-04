"""Manual test script: Parses a PDF paper and extracts claims using Gemma 4."""

import os
from mascv.rag.parsers.pdf_parser import PDFParser
from mascv.agents.claim_analyst import ClaimAnalystAgent
from mascv.core.state import InvestigationState
from mascv.utils.config_loader import load_config
from mascv.utils.llm import LLMClient


def main():
    # 1. Path to the sample LoRA paper
    pdf_path = os.path.join("data", "sample_inputs", "lora_2106.09685.pdf")

    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found at {pdf_path}")
        return

    print("=" * 75)
    print("STEP 1: Parsing PDF using PDFParser...")
    print("=" * 75)

    parser = PDFParser()
    paper = parser.parse(pdf_path)

    # Clean ligatures for Windows terminal printing
    clean_abstract = (paper.metadata.abstract or "").replace("\ufb01", "fi").replace("\ufb02", "fl")

    print(f"Title:         {paper.metadata.title}")
    print(f"arXiv ID:      {paper.metadata.arxiv_id}")
    print(f"Sections:      {len(paper.sections)} sections detected")
    print(f"Raw Text Len:  {len(paper.raw_text):,} characters")
    print(f"Abstract Snippet:\n  {clean_abstract[:250]}...")
    print("=" * 75)

    # 2. Connect Gemma 4 LLM Client and Claim Analyst
    print("\nSTEP 2: Initializing ClaimAnalystAgent with Gemma 4...")
    config = load_config("config/agents/claim_analyst.yaml")
    llm = LLMClient()

    agent = ClaimAnalystAgent(config=config, llm_client=llm)

    print(f"Agent Name:    {agent.name}")
    print(f"Connected Model: {llm.model_name}")
    print(f"Max Claims:    {agent.max_claims}")
    print(f"Allowed Types: {agent.allowed_types}")

    # 3. Create InvestigationState and Run Extraction
    print("\n" + "=" * 75)
    print("STEP 3: Running Gemma 4 claim extraction on the paper...")
    print("=" * 75)

    state = InvestigationState(paper=paper)
    updated_state = agent.execute(state)

    # 4. Display Extracted Claims
    print(f"\nSuccessfully extracted {len(updated_state.claims)} claims from LoRA!\n")
    for claim_id, claim_state in updated_state.claims.items():
        c = claim_state.claim
        print(f"[{c.id}] ({c.claim_type.upper()}) Subject: {c.subject}")
        print(f"  Statement:   {c.statement}")
        print(f"  Benchmarks:  {c.benchmarks}")
        print(f"  Metrics:     {c.metrics}")
        print(f"  Comparisons: {c.comparisons}")
        if c.conditions:
            print(f"  Conditions:  {c.conditions}")
        print("-" * 75)


if __name__ == "__main__":
    main()
