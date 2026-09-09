"""Claim Analyst Agent: Extracts and normalizes testable scientific claims."""

import json
from typing import Any, Dict, List, Optional
from mascv.agents.base import BaseAgent
from mascv.core.state import InvestigationState, ClaimInvestigationState
from mascv.models.claim import Claim, ClaimStatus
from mascv.utils.logger import get_logger
from mascv.utils.text_processing import extract_json_from_text

logger = get_logger(__name__)


class ClaimAnalystAgent(BaseAgent):
    """Extracts testable, meaningful propositions from target research papers."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        llm_client: Optional[Any] = None,
    ) -> None:
        if config is None:
            try:
                from mascv.utils.config_loader import load_config
                config = load_config("config/agents/claim_analyst.yaml")
            except Exception:
                config = {}
        super().__init__(name="ClaimAnalystAgent", config=config)

        agent_config = self.config.get("agent", {}) if self.config else {}
        self.model_name = agent_config.get("model", "gemma-4-26b-a4b-it")
        self.thinking_level = agent_config.get("thinking_level", "HIGH")
        self.temperature = agent_config.get("temperature", 0.2)

        from mascv.utils.llm import LLMClient
        self.llm_client = llm_client or LLMClient(
            model_name=self.model_name,
            temperature=self.temperature,
            thinking_level=self.thinking_level,
        )

        params = agent_config.get("parameters", {})
        self.max_claims = params.get("max_claims_to_extract", 10)
        self.allowed_types = params.get(
            "claim_types",
            ["performance", "efficiency", "causal", "generalization", "methodological", "novelty"],
        )
        self.require_quantitative_metrics = params.get("require_quantitative_metrics", False)

        # Load prompt templates for claim extraction
        prompts = self.config.get("prompts", {}) if self.config else {}
        self.system_prompt = prompts.get("system_prompt", "")
        self.user_prompt_template = prompts.get("user_prompt", "")

    def execute(self, state: InvestigationState) -> InvestigationState:
        """Process paper text and store structured claims in the investigation state."""
        paper_text = self._get_target_paper_text(state)
        paper_id = state.paper.id if state.paper and state.paper.id else "paper_1"

        extracted_claims = self.extract_claims(paper_text, paper_id=paper_id)

        for claim in extracted_claims:
            state.claims[claim.id] = ClaimInvestigationState(claim=claim)

        state.metadata["claims_extracted_count"] = len(extracted_claims)
        logger.info(f"ClaimAnalystAgent extracted {len(extracted_claims)} claim(s).")
        return state

    def _get_target_paper_text(self, state: InvestigationState) -> str:
        """Provide target paper text, respecting model context limits."""
        if not state.paper:
            return state.metadata.get("raw_paper_text", "")

        # Assemble from sections if available
        if state.paper.sections:
            parts: List[str] = []
            if state.paper.metadata and state.paper.metadata.abstract:
                parts.append(f"Abstract:\n{state.paper.metadata.abstract}")
            for s in state.paper.sections:
                parts.append(f"Section {s.title}:\n{s.content}")
            text = "\n\n".join(parts)
        elif state.paper.raw_text and state.paper.raw_text.strip():
            text = state.paper.raw_text
        else:
            text = state.metadata.get("raw_paper_text", "")

        # Guard against input token limit on Gemma 4 free tier (16,000 tokens ≈ 40,000 chars)
        max_chars = 18000
        if len(text) > max_chars:
            logger.info(f"Limiting paper text to first {max_chars:,} chars to respect model token quota.")
            text = text[:max_chars]

        return text

    def extract_claims(self, paper_text: str, paper_id: str = "paper_1") -> List[Claim]:
        """Formalize raw paper statements into structured Claim objects using Gemma 4."""
        if not paper_text or not paper_text.strip():
            logger.warning("ClaimAnalystAgent received empty paper text.")
            return []

        # 1. Format user prompt with parameters (using safe replace to avoid KeyError on JSON braces)
        clean_text = paper_text.replace("\ufb01", "fi").replace("\ufb02", "fl")
        if self.user_prompt_template:
            formatted_prompt = (
                self.user_prompt_template
                .replace("{paper_text}", clean_text)
                .replace("{max_claims}", str(self.max_claims))
                .replace("{allowed_types}", ", ".join(self.allowed_types))
            )
        else:
            formatted_prompt = f"Extract empirical claims from:\n{clean_text}"

        # 2. Call LLM client
        if self.llm_client is None:
            logger.warning("No LLM client configured for ClaimAnalystAgent.")
            return []

        try:
            if hasattr(self.llm_client, "generate"):
                response_text = self.llm_client.generate(
                    system_prompt=self.system_prompt,
                    prompt=formatted_prompt,
                )
            elif callable(self.llm_client):
                response_text = self.llm_client(formatted_prompt)
            else:
                logger.error("Unsupported LLM client interface.")
                return []
        except Exception as exc:
            logger.error(f"LLM claim extraction failed: {exc}")
            return []

        # 3. Clean and extract JSON substring (handles thought processes and markdown fences)
        json_str = extract_json_from_text(response_text)

        # 4. Parse JSON and validate into Pydantic Claim objects
        try:
            raw_data = json.loads(json_str)
            if isinstance(raw_data, dict) and "claims" in raw_data:
                raw_data = raw_data["claims"]
            elif isinstance(raw_data, dict):
                raw_data = [raw_data]

            if not isinstance(raw_data, list):
                logger.error(f"Expected JSON list of claims, got {type(raw_data)}.")
                return []

            claims: List[Claim] = []
            for idx, item in enumerate(raw_data, start=1):
                if not isinstance(item, dict):
                    continue

                item.setdefault("id", f"C{idx}")
                item.setdefault("paper_id", paper_id)
                item.setdefault("status", ClaimStatus.EXTRACTED)

                # Validate claim_type against allowed types
                c_type = item.get("claim_type", "performance")
                if c_type not in self.allowed_types:
                    item["claim_type"] = "performance"

                claim_obj = Claim.model_validate(item)

                # Filter if quantitative metrics are required
                if self.require_quantitative_metrics and not claim_obj.metrics:
                    continue

                claims.append(claim_obj)

            return claims[: self.max_claims]

        except Exception as exc:
            logger.error(
                f"Failed to parse LLM claims JSON: {exc}. Snippet: {response_text[:250]}"
            )
            return []
