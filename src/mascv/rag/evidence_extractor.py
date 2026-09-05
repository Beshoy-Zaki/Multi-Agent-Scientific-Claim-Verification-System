"""LLM-based extraction of scientific evidence."""

from typing import Any, List, Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

# Picks up GOOGLE_API_KEY / GEMINI_API_KEY from a .env file if present. Safe
# to call repeatedly / when no .env exists (it's a no-op in that case).
load_dotenv()


class ExtractedEvidence(BaseModel):
    """Structured result returned by the evidence extraction model."""

    relevant: bool = Field(
        description="Whether the text contains evidence relevant to the claim."
    )

    relationship: Literal[
        "SUPPORTS",
        "CONTRADICTS",
        "QUALIFIES",
        "REPLICATES",
        "CHALLENGES",
        "ALTERNATIVE",
    ]

    evidence_text: str

    context: str

    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
    )


class EvidenceExtractor:
    """Uses Gemini to classify and extract evidence."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        llm: Optional[Any] = None,
    ) -> None:
        """
        Args:
            model_name: Gemini model to use when ``llm`` is not supplied.
            llm: Optional pre-built chat model. Lets callers (and tests)
                inject a fake/mock LLM instead of requiring a live
                GOOGLE_API_KEY / GEMINI_API_KEY just to construct the agent.
        """

        if llm is not None:
            self.llm = llm
        else:
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.1,
                max_retries=2,
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

        prompt = f"""
You are a scientific evidence extraction component.

Scientific claim:
{claim}

Retrieved paper passage:
{chunk}

Determine whether this passage contains evidence
relevant to the claim.

Rules:

1. Do not use information outside the passage.
2. Do not invent numbers, experiments, authors, or conclusions.
3. If the passage is irrelevant, set relevant=false.
4. If it supports the claim, use SUPPORTS.
5. If it reports an independent replication, use REPLICATES.
6. If it weakens or disputes the claim, use CONTRADICTS or CHALLENGES.
7. If it limits the scope of the claim, use QUALIFIES.
8. Extract only evidence actually present in the passage.
9. Include important experimental context.
10. Confidence must be between 0 and 1.
"""

        return self.structured_llm.invoke(prompt)