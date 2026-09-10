"""Unit tests for PaperSearchAgent."""

import unittest
from unittest.mock import MagicMock
from mascv.agents.paper_search import PaperSearchAgent


class TestPaperSearchAgent(unittest.TestCase):
    """Test suite for PaperSearchAgent using search-grounded prompt templates."""

    def test_paper_search_initialization(self):
        agent = PaperSearchAgent()
        self.assertEqual(agent.name, "PaperSearchAgent")
        self.assertEqual(agent.max_papers, 5)
        self.assertIn("arxiv.org", agent.allowed_domains)
        self.assertIn("semanticscholar.org", agent.allowed_domains)
        self.assertTrue(len(agent.system_prompt) > 100)
        self.assertIn("{claim_statement}", agent.user_prompt_template)

    def test_paper_search_grounded_mock(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = """
        [
          {
            "title": "LoRA: Low-Rank Adaptation of Large Language Models",
            "authors": ["Edward J. Hu", "Yelong Shen", "Phillip Wallis"],
            "year": 2021,
            "venue": "ICLR 2022",
            "arxiv_id": "2106.09685",
            "url": "https://arxiv.org/abs/2106.09685",
            "abstract": "LoRA reduces trainable parameters by 10,000x and GPU memory requirements by 3x.",
            "key_findings": "Matches full fine-tuning performance on RoBERTa and DeBERTa with 0.01% parameters.",
            "relationship": "SUPPORTS",
            "relevance_score": 0.98,
            "relevance_rationale": "Direct seminal paper introducing the technique and evaluating GLUE benchmarks."
          },
          {
            "title": "On the Computational Limits of Low-Rank Adaptation",
            "authors": ["Adversarial Author"],
            "year": 2024,
            "venue": "arXiv preprint",
            "url": "https://arxiv.org/abs/2406.03136",
            "abstract": "We demonstrate that low rank r < 8 collapses representation diversity in complex reasoning.",
            "key_findings": "14% performance degradation on GSM8K compared to full parameter tuning.",
            "relationship": "CONTRADICTS",
            "relevance_score": 0.94,
            "relevance_rationale": "Demonstrates specific failure modes on math reasoning benchmarks."
          },
          {
            "title": "Random Tech Blog Post",
            "authors": ["Tech Blogger"],
            "year": 2024,
            "url": "https://medium.com/@dev/lora-is-awesome",
            "abstract": "An informal blog post praising LoRA without formal empirical rigor.",
            "relationship": "SUPPORTS"
          }
        ]
        """

        agent = PaperSearchAgent(llm_client=mock_llm)
        papers = agent.search_literature(
            claim_statement="LoRA reduces memory while maintaining performance on downstream tasks",
            claim_id="C1",
            claim_type="efficiency",
            benchmarks="GLUE, GSM8K",
        )

        # Verify Medium blog was filtered out and only the 2 arXiv papers were kept
        self.assertEqual(len(papers), 2)

        # First paper verification
        self.assertEqual(papers[0].title, "LoRA: Low-Rank Adaptation of Large Language Models")
        self.assertEqual(papers[0].arxiv_id, "2106.09685")
        self.assertEqual(papers[0].relationship, "SUPPORTS")
        self.assertEqual(papers[0].relevance_score, 0.98)
        self.assertIn("Edward J. Hu", papers[0].authors)

        # Second paper verification (testing regex fallback extraction of arxiv_id from url)
        self.assertEqual(papers[1].title, "On the Computational Limits of Low-Rank Adaptation")
        self.assertEqual(papers[1].arxiv_id, "2406.03136")
        self.assertEqual(papers[1].relationship, "CONTRADICTS")
        self.assertIn("14% performance degradation", papers[1].key_findings)

    def test_paper_search_filters_out_target_paper(self):
        """Programmatic rejection: Target paper candidates must be excluded to prevent self-validation."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = """
        [
          {
            "title": "LoRA: Low-Rank Adaptation of Large Language Models",
            "authors": ["Edward J. Hu", "Yelong Shen"],
            "year": 2021,
            "arxiv_id": "2106.09685",
            "url": "https://arxiv.org/abs/2106.09685",
            "relationship": "SUPPORTS"
          },
          {
            "title": "Independent Evaluation of Parameter-Efficient Fine-Tuning",
            "authors": ["Independent Researcher"],
            "year": 2023,
            "arxiv_id": "2305.12345",
            "url": "https://arxiv.org/abs/2305.12345",
            "relationship": "SUPPORTS"
          }
        ]
        """
        agent = PaperSearchAgent(llm_client=mock_llm)
        target_meta = {
            "title": "LoRA: Low-Rank Adaptation of Large Language Models",
            "arxiv_id": "2106.09685",
        }

        papers = agent.search_literature(
            claim_statement="LoRA reduces trainable parameters.",
            claim_id="C1",
            target_paper_info=target_meta,
        )

        # The target paper itself must be dropped; only the independent paper kept
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0].arxiv_id, "2305.12345")
        self.assertEqual(papers[0].title, "Independent Evaluation of Parameter-Efficient Fine-Tuning")
        self.assertEqual(papers[0].source_type, "EXTERNAL_SOURCE")
        self.assertTrue(papers[0].is_independent)



if __name__ == "__main__":
    unittest.main()
