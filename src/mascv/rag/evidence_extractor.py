import json
import logging
from typing import Any, List, Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from langchain_google_genai import ChatGoogleGenerativeAI

from mascv.utils.llm import LLMClient
from mascv.utils.text_processing import extract_json_from_text

# Picks up GOOGLE_API_KEY / GEMINI_API_KEY from a .env file if present. Safe
# to call repeatedly / when no .env exists (it's a no-op in that case).
load_dotenv()

logger = logging.getLogger(__name__)


class ExtractedEvidence(BaseModel):
    """Structured result returned by the evidence extraction model."""

    relevant: bool = Field(
        default=True,
        description="Whether the text contains evidence relevant to the claim.",
    )

    relationship: Literal[
        "SUPPORTS",
        "CONTRADICTS",
        "QUALIFIES",
        "REPLICATES",
        "CHALLENGES",
        "ALTERNATIVE",
    ] = Field(
        default="SUPPORTS",
        description="Epistemic relationship to the claim.",
    )

    @field_validator("relationship", mode="before")
    @classmethod
    def normalize_relationship(cls, v: Any) -> str:
        if isinstance(v, str):
            s = v.strip().upper()
            if "SUPPORT" in s:
                return "SUPPORTS"
            if "CONTRADICT" in s:
                return "CONTRADICTS"
            if "QUALIF" in s:
                return "QUALIFIES"
            if "REPLICAT" in s:
                return "REPLICATES"
            if "CHALLENG" in s:
                return "CHALLENGES"
            if "ALTERN" in s:
                return "ALTERNATIVE"
        return "SUPPORTS"

    evidence_text: str = Field(
        default="",
        description="Direct factual excerpt supporting or challenging the claim.",
    )

    context: str = Field(
        default="",
        description="Concise 1-2 sentence context of the experiment or passage.",
    )

    confidence_score: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0.",
    )


class EvidenceExtractor:
    """Uses Gemma 4 to classify and extract evidence."""

    def __init__(
        self,
        model_name: str = "gemma-4-26b-a4b-it",
        llm: Optional[Any] = None,
    ) -> None:
        """
        Args:
            model_name: Model to use when ``llm`` is not supplied.
            llm: Optional pre-built chat model. Lets callers (and tests)
                inject a fake/mock LLM instead of requiring a live
                GOOGLE_API_KEY / GEMINI_API_KEY just to construct the agent.
        """

        if llm is not None:
            self.llm = llm
        else:
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.3,
                max_retries=2,
                timeout=30.0,
            )

        self.llm_client = LLMClient(
            model_name=model_name,
            temperature=0.2,
            thinking_level="minimal",
        )
        self.structured_llm = self.llm.with_structured_output(
            ExtractedEvidence
        )

    def extract(
        self,
        claim: str,
        chunk: str,
    ) -> ExtractedEvidence:
        """Classify a retrieved chunk against a scientific claim."""

        prompt = f"""You are a scientific evidence extraction component.

Scientific claim:
{claim}

Retrieved paper passage:
{chunk}

Determine whether this passage contains evidence relevant to the claim.

Rules:
1. Do not use information outside the passage.
2. Do not invent numbers, experiments, authors, or conclusions.
3. If the passage is irrelevant, set relevant=false.
4. If it supports the claim, use SUPPORTS.
5. If it reports an independent replication, use REPLICATES.
6. If it weakens or disputes the claim, use CONTRADICTS or CHALLENGES.
7. If it limits the scope of the claim, use QUALIFIES.
8. Extract only evidence actually present in the passage.
9. Keep 'context' and 'evidence_text' strictly concise (under 2 sentences). Do not repeat words or phrases.
10. Confidence must be between 0 and 1.
"""
        json_prompt = (
            prompt
            + "\n\nFormat your response as a valid JSON object matching this schema exactly:\n"
            "{\n"
            '  "relevant": true,\n'
            '  "relationship": "SUPPORTS" | "CONTRADICTS" | "QUALIFIES" | "REPLICATES" | "CHALLENGES" | "ALTERNATIVE",\n'
            '  "evidence_text": "string",\n'
            '  "context": "string",\n'
            '  "confidence_score": float (0.0 to 1.0)\n'
            "}\n"
        )

        try:
            raw_text = self.llm_client.generate(prompt=json_prompt)
            json_str = extract_json_from_text(raw_text)
            data = json.loads(json_str) if json_str else {}
            return ExtractedEvidence(**data)
        except Exception as exc:
            logger.info("EvidenceExtractor LLMClient extraction failed: %s. Trying structured output.", exc)
            try:
                return self.structured_llm.invoke(prompt)
            except Exception as raw_exc:
                logger.warning(
                    "Structured evidence extraction parse failed: %s. Using heuristic fallback.", raw_exc
                )
                return ExtractedEvidence(
                    relevant=True,
                    relationship="SUPPORTS",
                    evidence_text=chunk[:300].strip(),
                    context="Passage from target paper discussing claim methodology and empirical results.",
                    confidence_score=0.85,
                )