"""Paper Search Agent: Grounded literature discovery agent using Gemma 4 with live web search."""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from mascv.agents.base import BaseAgent
from mascv.models.paper import PaperMetadata
from mascv.utils.config_loader import load_config
from mascv.utils.llm import LLMClient
from mascv.utils.text_processing import extract_json_from_text

logger = logging.getLogger(__name__)


class PaperSearchAgent(BaseAgent):
    """Discovers real academic literature using Gemma 4 with live Google Search Grounding.
    
    Replaces complex, multi-file Python API wrappers (arXiv, Semantic Scholar, CrossRef)
    with an intelligent, prompt-driven search-grounded discovery engine.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        llm_client: Optional[Any] = None,
    ) -> None:
        """Initialize agent with configuration and search-grounded Gemma 4 client."""
        if config is None:
            try:
                config = load_config("config/agents/paper_search.yaml")
            except Exception:
                config = {}
        super().__init__(name="PaperSearchAgent", config=config)

        # Agent parameters from YAML
        agent_config = self.config.get("agent", {}) if self.config else {}
        self.params = agent_config.get("parameters", {}) or self.config.get("parameters", {})
        self.max_papers = self.params.get("max_papers", 5)
        self.allowed_domains = self.params.get(
            "allowed_domains",
            [
                "arxiv.org",
                "semanticscholar.org",
                "doi.org",
                "ncbi.nlm.nih.gov",
                "openreview.net",
                "aclanthology.org",
                "nature.com",
                "ieee.org",
            ],
        )

        # Load prompts from config/agents/paper_search.yaml
        prompts = self.config.get("prompts", {}) if self.config else {}
        self.system_prompt = prompts.get("system_prompt", "")
        self.user_prompt_template = prompts.get("user_prompt", "")

        self.thinking_level = agent_config.get("thinking_level", "minimal")

        # Initialize Gemma 4 LLM client with grounding enabled and high thinking level
        self.llm_client = llm_client or LLMClient(
            model_name="gemma-4-26b-a4b-it",
            temperature=0.1,
            enable_grounding=True,
            thinking_level=self.thinking_level,
        )

    def execute(self, state: Any) -> Any:
        """Execute grounded literature search for the active claim in the investigation state."""
        claims = getattr(state, "claims", {}) if hasattr(state, "claims") else state.get("claims", {})
        active_claim_id = (
            getattr(state, "active_claim_id", None)
            if hasattr(state, "active_claim_id")
            else state.get("active_claim_id")
        )

        claim_obj = None
        if active_claim_id and active_claim_id in claims:
            claim_state = claims[active_claim_id]
            claim_obj = getattr(claim_state, "claim", claim_state)
        elif claims:
            first_key = next(iter(claims))
            claim_state = claims[first_key]
            claim_obj = getattr(claim_state, "claim", claim_state)

        statement = ""
        claim_id = active_claim_id or "C1"
        claim_type = "performance"
        subject = "Scientific / Empirical Claim"
        benchmarks = ""

        if claim_obj:
            statement = getattr(claim_obj, "statement", str(claim_obj))
            claim_id = getattr(claim_obj, "id", claim_id)
            claim_type = getattr(claim_obj, "claim_type", claim_type)
            subject = getattr(claim_obj, "subject", subject)
            benchmarks_list = getattr(claim_obj, "benchmarks", [])
            metrics_list = getattr(claim_obj, "metrics", [])
            all_metrics = list(benchmarks_list) + list(metrics_list)
            if all_metrics:
                benchmarks = ", ".join(all_metrics)
        elif isinstance(state, dict):
            statement = state.get("claim_statement", "")
            subject = state.get("subject", subject)
            benchmarks = state.get("benchmarks", "")

        logger.info("Executing Gemma 4 Grounded PaperSearch for: '%s'", statement[:80])

        # Extract target paper metadata to enforce target paper exclusion
        target_paper = getattr(state, "paper", None) if hasattr(state, "paper") else (state.get("paper") if isinstance(state, dict) else None)
        target_meta = getattr(target_paper, "metadata", None) if target_paper else None
        target_paper_info = {}
        if target_meta:
            target_paper_info = {
                "title": getattr(target_meta, "title", "") or "",
                "arxiv_id": getattr(target_meta, "arxiv_id", "") or "",
                "doi": getattr(target_meta, "doi", "") or "",
                "authors": getattr(target_meta, "authors", []) or [],
            }
        elif isinstance(target_paper, dict):
            p_meta = target_paper.get("metadata", {}) or {}
            target_paper_info = {
                "title": p_meta.get("title") or target_paper.get("title") or "",
                "arxiv_id": p_meta.get("arxiv_id") or target_paper.get("arxiv_id") or "",
                "doi": p_meta.get("doi") or target_paper.get("doi") or "",
                "authors": p_meta.get("authors") or target_paper.get("authors") or [],
            }

        # Discover papers using Gemma 4 with live web grounding
        discovered_papers = self.search_literature(
            claim_statement=statement,
            claim_id=claim_id,
            claim_type=str(claim_type),
            subject=str(subject),
            benchmarks=benchmarks,
            target_paper_info=target_paper_info,
        )

        # Store discovered papers in state (preserving both titles and full metadata)
        if hasattr(state, "claims") and active_claim_id in state.claims:
            target_claim_state = state.claims[active_claim_id]
            if hasattr(target_claim_state, "external_papers_found"):
                target_claim_state.external_papers_found.extend(
                    [p.title for p in discovered_papers]
                )
            if hasattr(target_claim_state, "discovered_papers_metadata"):
                target_claim_state.discovered_papers_metadata.extend(discovered_papers)

            # Store in global state metadata for RAG/debate agent retrieval
            if hasattr(state, "metadata") and isinstance(state.metadata, dict):
                state.metadata.setdefault("discovered_papers_metadata", {})[active_claim_id] = discovered_papers
        elif isinstance(state, dict):
            if "discovered_papers" not in state:
                state["discovered_papers"] = []
            state["discovered_papers"].extend(discovered_papers)

        return state

    def search_literature(
        self,
        claim_statement: str,
        claim_id: str = "C1",
        claim_type: str = "performance",
        subject: Optional[str] = None,
        benchmarks: Optional[str] = None,
        adversarial_focus: Optional[str] = None,
        target_paper_info: Optional[Dict[str, Any]] = None,
    ) -> List[PaperMetadata]:
        """Perform prompt-driven academic search across arXiv, Semantic Scholar, PubMed, etc.
        
        Substitutes all custom API client scrapers with a single search-grounded call.
        """
        if not claim_statement or not claim_statement.strip():
            return []

        clean_subject = subject or "Scientific Empirical Methodology"
        clean_adversarial = (
            adversarial_focus
            or "Empirical limitations, negative replications, benchmark contamination, scaling failure modes"
        )

        # 1. Format user prompt with claim specifics
        formatted_prompt = (
            self.user_prompt_template
            .replace("{claim_id}", str(claim_id))
            .replace("{claim_statement}", claim_statement)
            .replace("{claim_type}", str(claim_type))
            .replace("{subject}", clean_subject)
            .replace("{adversarial_focus}", clean_adversarial)
            .replace("{max_papers}", str(self.max_papers))
        )

        if target_paper_info and target_paper_info.get("title"):
            t_title_val = target_paper_info.get("title", "")
            t_auth_val = ", ".join(target_paper_info.get("authors", []))
            target_context = (
                f"\n=== TARGET RESEARCH PAPER (UNDER INVESTIGATION) ===\n"
                f"Title: {t_title_val}\n"
                f"Authors: {t_auth_val}\n"
                f"Identifier: {target_paper_info.get('arxiv_id') or target_paper_info.get('doi') or 'N/A'}\n"
                f"CRITICAL MANDATE: You are investigating claims made by THIS paper. You MUST NOT return this target paper as an external source. "
                f"All retrieved literature MUST be INDEPENDENT external work from other research groups.\n"
                f"====================================================\n\n"
            )
            formatted_prompt = target_context + formatted_prompt

        # Dynamically omit benchmarks if empty or generic
        clean_bench = str(benchmarks or "").strip()
        is_generic_benchmarks = (
            not clean_bench
            or clean_bench.lower() in ["standard benchmarks", "none", "n/a", "null"]
        )

        if is_generic_benchmarks:
            # Omit the benchmarks line so search targets the method and technique directly
            formatted_prompt = formatted_prompt.replace(
                "Key Benchmarks / Metrics: {benchmarks_and_metrics}\n", ""
            )
            formatted_prompt = formatted_prompt.replace(
                "Key Benchmarks / Metrics: {benchmarks_and_metrics}", ""
            )
            formatted_prompt = formatted_prompt.replace(
                "verifying the stated metrics on the specified benchmarks.",
                "verifying the stated findings and methodology.",
            )
        else:
            formatted_prompt = formatted_prompt.replace(
                "{benchmarks_and_metrics}", clean_bench
            )

        # 2. Call Gemma 4 with live Google Search Grounding
        try:
            logger.info("Calling Gemma 4 a4b with Google Search Grounding...")
            response_text = self.llm_client.generate(
                prompt=formatted_prompt,
                system_prompt=self.system_prompt,
                enable_grounding=True,
            )

            # 3. Extract JSON from the model's response
            json_str = extract_json_from_text(response_text)
            if not json_str:
                logger.warning("No JSON found in Gemma 4 search response: %s", response_text[:200])
                return []

            raw_papers = json.loads(json_str)
            if isinstance(raw_papers, dict):
                raw_papers = (
                    raw_papers.get("papers")
                    or raw_papers.get("discovered_papers")
                    or raw_papers.get("publications")
                    or [raw_papers]
                )

            # Target paper normalization for programmatic exclusion filter
            t_norm_title = re.sub(r"[^a-z0-9]", "", target_paper_info.get("title", "").lower()) if target_paper_info else ""
            t_arxiv = str(target_paper_info.get("arxiv_id", "") or "").strip().lower() if target_paper_info else ""
            t_doi = str(target_paper_info.get("doi", "") or "").strip().lower() if target_paper_info else ""

            # 4. Domain safety filter & PaperMetadata conversion
            papers: List[PaperMetadata] = []
            for item in raw_papers:
                if not isinstance(item, dict):
                    continue

                url = str(item.get("url") or "").strip()
                # Ensure the paper URL originates from allowed academic repositories
                if self.allowed_domains and not any(d in url.lower() for d in self.allowed_domains):
                    logger.debug("Filtered out non-academic URL: %s", url)
                    continue

                # Automatically extract arXiv ID from URL if missing
                arxiv_id = item.get("arxiv_id")
                if not arxiv_id and "arxiv.org" in url:
                    match = re.search(r"(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}|[a-zA-Z-]+/[0-9]+)", url)
                    if match:
                        arxiv_id = match.group(1)

                # Automatically extract DOI from URL if missing
                doi = item.get("doi")
                if not doi and "doi.org" in url:
                    match = re.search(r"doi\.org/(10\.[0-9]{4,9}/[-._;()/:A-Za-z0-9]+)", url)
                    if match:
                        doi = match.group(1)

                # Programmatic target-paper exclusion to prevent circular self-support
                item_title = str(item.get("title") or "")
                norm_item_title = re.sub(r"[^a-z0-9]", "", item_title.lower())
                if t_norm_title and len(t_norm_title) > 6:
                    if norm_item_title == t_norm_title or (len(norm_item_title) > 10 and (t_norm_title in norm_item_title or norm_item_title in t_norm_title)):
                        logger.info("Excluding target paper from external search results by title: '%s'", item_title)
                        continue

                if t_arxiv and arxiv_id and (t_arxiv in arxiv_id.lower() or arxiv_id.lower() in t_arxiv):
                    logger.info("Excluding target paper from external search results by arXiv ID: '%s'", arxiv_id)
                    continue

                if t_doi and doi and (t_doi in doi.lower() or doi.lower() in t_doi):
                    logger.info("Excluding target paper from external search results by DOI: '%s'", doi)
                    continue

                metadata = PaperMetadata(
                    title=item.get("title", "Untitled Publication"),
                    authors=item.get("authors") or [],
                    abstract=item.get("abstract"),
                    arxiv_id=arxiv_id,
                    doi=doi,
                    url=url,
                    year=item.get("year"),
                    venue=item.get("venue") or "Academic Repository",
                    relationship=item.get("relationship"),
                    relevance_score=item.get("relevance_score"),
                    relevance_rationale=item.get("relevance_rationale"),
                    key_findings=item.get("key_findings"),
                    source_type="EXTERNAL_SOURCE",
                    is_independent=True,
                )
                papers.append(metadata)

            return papers[: self.max_papers]

        except Exception as e:
            logger.error("Gemma 4 grounded search failed: %s", e)
            return []
