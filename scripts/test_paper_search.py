"""Manual test script to verify Search-Grounded PaperSearchAgent with Gemma 4."""

import os
import sys

# Ensure src is in python path
sys.path.insert(0, "src")

from dotenv import load_dotenv
from mascv.agents.paper_search import PaperSearchAgent
from mascv.utils.llm import LLMClient

load_dotenv()


def main():
    print("=" * 80)
    print("STEP 1: Initializing Gemma 4 Search-Grounded Agent...")
    print("=" * 80)

    # Initialize Gemma 4 a4b with Google Search Grounding enabled
    llm = LLMClient(model_name="gemma-4-26b-a4b-it", temperature=0.1, enable_grounding=True)
    agent = PaperSearchAgent(llm_client=llm)

    print(f"Agent Name:       {agent.name}")
    print(f"Model:            {llm.model_name}")
    print(f"Target Domains:   {agent.allowed_domains}")
    print(f"Max Papers:       {agent.max_papers}")

    claim = "LoRA reduces memory while maintaining performance on downstream tasks"
    print(f"\nTarget Claim:     '{claim}'\n")

    print("=" * 80)
    print("STEP 2: Executing Live Grounded Search via Gemma 4 Prompt...")
    print("(Gemma 4 searches Google live targeting site:arxiv.org, site:semanticscholar.org, etc.)")
    print("=" * 80)

    papers = agent.search_literature(
        claim_statement=claim,
        claim_id="C1",
        claim_type="efficiency",
        subject="Parameter-Efficient Fine-Tuning (PEFT)",
        benchmarks="GLUE, SuperGLUE, SQuAD",
        adversarial_focus="Limitations on complex reasoning tasks, memory overhead, low-rank collapse",
    )

    print(f"\nSuccessfully discovered {len(papers)} candidate papers:\n")
    for i, p in enumerate(papers, 1):
        print(f"[{i}] {p.title}")
        print(f"    Authors:       {', '.join(p.authors[:4])}")
        print(f"    Venue/Year:    {p.venue} ({p.year})")
        print(f"    arXiv ID:      {p.arxiv_id or 'N/A'}")
        print(f"    DOI:           {p.doi or 'N/A'}")
        print(f"    URL:           {p.url}")
        print(f"    Relationship:  {p.relationship} (Relevance: {p.relevance_score})")
        if p.key_findings:
            print(f"    Key Findings:  {p.key_findings}")
        if p.abstract:
            print(f"    Abstract:      {p.abstract[:220]}...")
        print("-" * 80)


if __name__ == "__main__":
    main()
