"""Manual test script to verify PaperSearchAgent using Gemma 4 and direct academic search."""

import os
from dotenv import load_dotenv
from mascv.agents.paper_search import PaperSearchAgent
from mascv.utils.llm import LLMClient

load_dotenv()


def main():
    print("=" * 70)
    print("STEP 1: Initializing PaperSearchAgent & Gemma 4 LLM...")
    print("=" * 70)

    # Initialize Gemma 4 LLMClient
    llm = LLMClient(model_name="gemma-4-31b-it", temperature=0.4)
    agent = PaperSearchAgent(llm_client=llm)

    print(f"Agent Name:    {agent.name}")
    print(f"Model:         {llm.model_name}")
    print(f"Engines:       {[c.__class__.__name__ for c in agent.search_clients]}")

    claim = "LoRA achieves comparable performance to full fine-tuning with fewer trainable parameters"
    print(f"\nTarget Claim:  '{claim}'\n")

    print("=" * 70)
    print("STEP 2: Generating Paired Adversarial Queries via Gemma 4...")
    print("=" * 70)

    queries = agent.generate_adversarial_queries(
        claim_statement=claim,
        claim_id="C1",
        claim_type="efficiency",
        benchmarks="GLUE, SuperGLUE, RoBERTa, DeBERTa",
    )

    print("\nSupporting Queries (Confirming / Replications):")
    for q in queries.get("supporting_queries", []):
        print(f"  [+] {q}")

    print("\nAdversarial Queries (Refuting / Limitations / Failure Cases):")
    for q in queries.get("adversarial_queries", []):
        print(f"  [-] {q}")

    print("\n" + "=" * 70)
    print("STEP 3: Searching Academic Literature (arXiv, Semantic Scholar, CrossRef)...")
    print("=" * 70)

    test_queries = queries.get("supporting_queries", [])[:1] + queries.get("adversarial_queries", [])[:1]
    papers = agent.search_literature(test_queries)

    print(f"\nSuccessfully discovered {len(papers)} candidate papers:\n")
    for i, p in enumerate(papers, 1):
        print(f"[{i}] {p.title}")
        print(f"    Authors:  {', '.join(p.authors[:3])}")
        print(f"    Venue:    {p.venue} ({p.year})")
        print(f"    URL:      {p.url or p.arxiv_id or p.doi}")
        if p.abstract:
            print(f"    Snippet:  {p.abstract[:180]}...")
        print("-" * 70)


if __name__ == "__main__":
    main()
