"""Paper Search Agent: Discovers external literature via adversarial search queries and academic engines."""

import json
import logging
from typing import Any, Dict, List, Optional

from mascv.agents.base import BaseAgent
from mascv.models.paper import PaperMetadata
from mascv.search.academic_search.arxiv_client import ArxivClient
from mascv.search.academic_search.crossref_client import CrossrefClient
from mascv.search.academic_search.semantic_scholar_client import SemanticScholarClient
from mascv.search.base_search import BaseSearchClient
from mascv.utils.config_loader import load_config
from mascv.utils.llm import LLMClient
from mascv.utils.text_processing import extract_json_from_text

logger = logging.getLogger(__name__)


class PaperSearchAgent(BaseAgent):
    """Executes multi-query adversarial searches across academic engines using Gemma 4."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        llm_client: Optional[Any] = None,
    ) -> None:
        """Initialize search agent with search clients, engine parameters, and LLM client."""
        if config is None:
            try:
                config = load_config("config/agents/paper_search.yaml")
            except Exception:
                config = {}
        super().__init__(name="PaperSearchAgent", config=config)

        # Agent parameters
        agent_config = self.config.get("agent", {}) if self.config else {}
        self.params = agent_config.get("parameters", {}) or self.config.get("parameters", {})
        self.max_queries_per_side = self.params.get("max_queries_per_side", 4)
        self.max_papers_per_claim = self.params.get("max_papers_per_claim", 5)
        self.enable_adversarial = self.params.get("enable_adversarial_queries", True)

        # Load prompts from config
        prompts = self.config.get("prompts", {}) if self.config else {}
        self.system_prompt = prompts.get("system_prompt", "")
        self.user_prompt_template = prompts.get("user_prompt", "")

        # LLM Client (defaults to Gemma 4 from .env)
        self.llm_client = llm_client

        # Academic search engines (100% Free, zero search API keys needed)
        engine_names = self.params.get(
            "search_engines", ["arxiv", "semantic_scholar", "crossref"]
        )
        self.search_clients: List[BaseSearchClient] = []

        if "arxiv" in engine_names:
            self.search_clients.append(ArxivClient())
        if "semantic_scholar" in engine_names:
            self.search_clients.append(SemanticScholarClient())
        if "crossref" in engine_names:
            self.search_clients.append(CrossrefClient())

        if not self.search_clients:
            self.search_clients.append(ArxivClient())

    def execute(self, state: Any) -> Any:
        """Discover candidate external papers for the active claim in the investigation state."""
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
        benchmarks = "Standard benchmarks"

        if claim_obj:
            statement = getattr(claim_obj, "statement", str(claim_obj))
            claim_id = getattr(claim_obj, "id", claim_id)
            claim_type = getattr(claim_obj, "claim_type", claim_type)
            benchmarks_list = getattr(claim_obj, "benchmarks", [])
            if benchmarks_list:
                benchmarks = ", ".join(benchmarks_list)
        elif isinstance(state, dict):
            statement = state.get("claim_statement", "")

        logger.info("Running PaperSearchAgent for claim: '%s'", statement[:80])

        # 1. Formulate supporting & adversarial queries using Gemma 4
        queries_dict = self.generate_adversarial_queries(
            claim_statement=statement,
            claim_id=claim_id,
            claim_type=str(claim_type),
            benchmarks=benchmarks,
        )
        all_queries = (
            queries_dict.get("supporting_queries", [])
            + queries_dict.get("adversarial_queries", [])
        )

        # 2. Search academic engines (arXiv, Semantic Scholar, CrossRef)
        discovered_papers = self.search_literature(all_queries)

        # 3. Store discovered papers in state
        if hasattr(state, "claims") and active_claim_id in state.claims:
            target_claim_state = state.claims[active_claim_id]
            if hasattr(target_claim_state, "external_papers_found"):
                target_claim_state.external_papers_found.extend(
                    [p.title for p in discovered_papers]
                )
        elif isinstance(state, dict):
            if "discovered_papers" not in state:
                state["discovered_papers"] = []
            state["discovered_papers"].extend(discovered_papers)
            state["search_queries"] = queries_dict

        return state

    def generate_adversarial_queries(
        self,
        claim_statement: str,
        claim_id: str = "C1",
        claim_type: str = "performance",
        benchmarks: str = "Standard benchmarks",
    ) -> Dict[str, List[str]]:
        """Formulate paired supporting and adversarial queries using Gemma 4 and prompt templates."""
        if not claim_statement or not claim_statement.strip():
            return {"supporting_queries": [], "adversarial_queries": []}

        # Initialize LLM client if not already provided
        client = self.llm_client
        if client is None:
            try:
                client = LLMClient(model_name="gemma-4-31b-it", temperature=0.4)
            except Exception as e:
                logger.debug("Could not auto-initialize LLMClient: %s", e)

        # 1. Use Gemma 4 with prompt template
        if client and self.user_prompt_template:
            formatted_prompt = (
                self.user_prompt_template
                .replace("{claim_id}", str(claim_id))
                .replace("{claim_statement}", claim_statement)
                .replace("{claim_type}", str(claim_type))
                .replace("{benchmarks_and_metrics}", str(benchmarks))
                .replace("{max_queries_per_side}", str(self.max_queries_per_side))
            )

            try:
                logger.info("Generating adversarial queries with Gemma 4...")
                if hasattr(client, "generate"):
                    response_text = client.generate(
                        prompt=formatted_prompt,
                        system_prompt=self.system_prompt,
                    )
                elif callable(client):
                    response_text = client(formatted_prompt)
                else:
                    response_text = ""

                json_str = extract_json_from_text(response_text)
                if json_str:
                    data = json.loads(json_str)
                    search_queries = data.get("search_queries", data)
                    sup = search_queries.get("supporting_queries", [])
                    adv = search_queries.get("adversarial_queries", [])
                    if sup or adv:
                        return {
                            "supporting_queries": sup[: self.max_queries_per_side],
                            "adversarial_queries": adv[: self.max_queries_per_side],
                        }
            except Exception as e:
                logger.warning("LLM query generation failed, falling back to heuristic: %s", e)

        # 2. Heuristic fallback when LLM is unavailable
        return self._heuristic_queries(claim_statement)

    def _heuristic_queries(self, claim_statement: str) -> Dict[str, List[str]]:
        """Deterministic keyword heuristic for fallback query generation."""
        clean_stmt = claim_statement.strip().rstrip(".")

        supporting = [
            f'"{clean_stmt}"',
            f"{clean_stmt} replication",
            f"{clean_stmt} benchmark",
        ][: self.max_queries_per_side]

        adversarial = []
        if self.enable_adversarial:
            adversarial = [
                f"{clean_stmt} limitations",
                f"{clean_stmt} failure cases",
                f"{clean_stmt} benchmark contamination",
                f"{clean_stmt} baseline discrepancy",
            ][: self.max_queries_per_side]

        return {
            "supporting_queries": supporting,
            "adversarial_queries": adversarial,
        }

    def search_literature(self, queries: List[str]) -> List[PaperMetadata]:
        """Query academic search engines and deduplicate candidate papers."""
        seen_titles = set()
        seen_ids = set()
        all_papers: List[PaperMetadata] = []

        limit_per_query = max(1, self.max_papers_per_claim // max(1, len(queries)))

        for query in queries:
            for client in self.search_clients:
                try:
                    papers = client.search(query, max_results=limit_per_query)
                    for paper in papers:
                        norm_title = paper.title.lower().strip()
                        paper_id = paper.arxiv_id or paper.doi or norm_title

                        if norm_title in seen_titles or paper_id in seen_ids:
                            continue

                        seen_titles.add(norm_title)
                        seen_ids.add(paper_id)
                        all_papers.append(paper)

                        if len(all_papers) >= self.max_papers_per_claim:
                            return all_papers
                except Exception as e:
                    logger.warning("Search client error on query '%s': %s", query, e)

        return all_papers
