"""RAG / Evidence Agent."""

from typing import Any, Dict, List
from uuid import uuid4

from langchain.tools import tool

from mascv.agents.base import BaseAgent
from mascv.models.evidence import (
    EvidenceBundle,
    EvidenceRelationship,
)

from mascv.rag.document_parser import parse_pdf
from mascv.rag.retriever import HybridRetriever
from mascv.rag.retriever.reranker import rerank_chunks
from mascv.rag.evidence_extractor import EvidenceExtractor


@tool
def search_evidence(
    claim: str,
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Retrieve candidate evidence passages relevant to a scientific claim."""

    retriever = HybridRetriever()

    return retriever.retrieve(
        query=claim,
        chunks=chunks,
        top_k=10,
    )


@tool
def rerank_evidence(
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Rerank retrieved scientific evidence candidates."""

    return rerank_chunks(
        chunks,
        top_k=5,
    )


class EvidenceRAGAgent(BaseAgent):
    """
    Extracts claim-aware evidence from target
    and external scientific papers.
    """

    def __init__(
        self,
        config: Dict[str, Any] = None,
        retriever: Any = None,
        extractor: Any = None,
    ) -> None:
        """
        Args:
            config: Agent configuration (flat dict or a loaded
                ``config/agents/evidence_rag.yaml`` with an ``agent`` key).
            retriever: Optional pre-built retriever (defaults to a fresh
                ``HybridRetriever``). Useful for tests.
            extractor: Optional pre-built ``EvidenceExtractor`` (or a mock).
                Lets callers inject a fake extractor instead of requiring a
                live GOOGLE_API_KEY / GEMINI_API_KEY just to construct the
                agent.
        """

        super().__init__(
            name="EvidenceRAGAgent",
            config=config,
        )

        self.retriever = retriever or HybridRetriever()

        self.extractor = extractor or EvidenceExtractor(
            model_name=self.config.get(
                "model",
                "gemini-2.5-flash",
            )
        )

    def execute(
        self,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run the evidence pipeline for the active claim.
        """

        claim_state = self._get_active_claim(state)

        claim = claim_state["claim"]

        claim_text = claim["statement"]

        papers = self._get_papers(state)

        all_evidence = []

        for paper in papers:

            chunks = self._create_chunks(paper)

            if not chunks:
                continue

            retrieved = self.retriever.retrieve(
                query=claim_text,
                chunks=chunks,
                top_k=10,
            )

            ranked = rerank_chunks(
                retrieved,
                top_k=5,
            )

            for chunk in ranked:

                extracted = self.extractor.extract(
                    claim=claim_text,
                    chunk=chunk["text"],
                )

                if not extracted.relevant:
                    continue

                evidence = self._build_bundle(
                    claim=claim,
                    paper=paper,
                    chunk=chunk,
                    extracted=extracted,
                )

                all_evidence.append(evidence)

        evidence_store = state.setdefault(
            "global_evidence_store",
            {},
        )

        for evidence in all_evidence:
            evidence_store[evidence.id] = evidence.model_dump()

        claim_state["evidence_bundle_ids"] = [
            evidence.id
            for evidence in all_evidence
        ]

        claim_state["status_message"] = (
            f"Extracted {len(all_evidence)} evidence bundles."
        )

        # `_get_active_claim` returns a fresh dict (via `model_dump()`) when
        # `state["claims"][active_claim_id]` is a pydantic
        # `ClaimInvestigationState`, which is the canonical shape defined in
        # `mascv.core.state`. Without writing it back explicitly here, every
        # mutation above (evidence_bundle_ids, status_message) is silently
        # lost as soon as the input state uses real pydantic models instead
        # of plain dicts.
        state["claims"][state["active_claim_id"]] = claim_state

        return state

    def _get_active_claim(
        self,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:

        active_claim_id = state.get(
            "active_claim_id"
        )

        if not active_claim_id:
            raise ValueError(
                "state.active_claim_id is missing."
            )

        claims = state.get("claims", {})

        if active_claim_id not in claims:
            raise ValueError(
                f"Claim {active_claim_id} not found in state."
            )

        claim_state = claims[active_claim_id]

        if hasattr(claim_state, "model_dump"):
            return claim_state.model_dump()

        return claim_state

    def _get_papers(
        self,
        state: Dict[str, Any],
    ) -> List[Dict[str, Any]]:

        papers = []

        target_paper = state.get("paper")

        if target_paper:

            if hasattr(target_paper, "model_dump"):
                papers.append(
                    target_paper.model_dump()
                )
            else:
                papers.append(target_paper)

        external_papers = (
            state.get("metadata", {})
            .get("external_papers", [])
        )

        for paper in external_papers:

            if hasattr(paper, "model_dump"):
                papers.append(
                    paper.model_dump()
                )
            else:
                papers.append(paper)

        return papers

    def _create_chunks(
        self,
        paper: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Convert paper content into retrieval chunks.
        """

        chunks = []

        sections = paper.get(
            "sections",
            [],
        )

        if sections:

            for section in sections:

                section_title = section.get(
                    "title",
                    "Unknown",
                )

                content = section.get(
                    "content",
                    "",
                )

                page_number = section.get(
                    "page_number",
                )

                words = content.split()

                chunk_size = 350
                overlap = 50

                start = 0

                while start < len(words):

                    end = min(
                        start + chunk_size,
                        len(words),
                    )

                    text = " ".join(
                        words[start:end]
                    )

                    if text.strip():

                        chunks.append(
                            {
                                "text": text,
                                "section": section_title,
                                "page_number": page_number,
                                "paper_id": paper["id"],
                            }
                        )

                    if end == len(words):
                        break

                    start = end - overlap

        else:

            raw_text = paper.get(
                "raw_text",
                "",
            )

            words = raw_text.split()

            for start in range(
                0,
                len(words),
                300,
            ):

                end = min(
                    start + 300,
                    len(words),
                )

                chunks.append(
                    {
                        "text": " ".join(
                            words[start:end]
                        ),
                        "section": "Unknown",
                        "page_number": None,
                        "paper_id": paper["id"],
                    }
                )

        return chunks

    def _build_bundle(
        self,
        claim: Dict[str, Any],
        paper: Dict[str, Any],
        chunk: Dict[str, Any],
        extracted: Any,
    ) -> EvidenceBundle:

        page = chunk.get(
            "page_number"
        )

        if page:
            location = f"Page {page}"
        else:
            location = (
                f"Section {chunk.get('section', 'Unknown')}"
            )

        return EvidenceBundle(
            id=f"E-{uuid4().hex[:8]}",
            claim_id=claim["id"],
            source_paper_id=paper["id"],
            source_title=paper.get(
                "metadata",
                {},
            ).get(
                "title",
                paper["id"],
            ),
            location=location,
            content=extracted.evidence_text,
            context=extracted.context,
            relationship=EvidenceRelationship(
                extracted.relationship
            ),
            confidence_score=extracted.confidence_score,
        )

    def assemble_evidence_bundle(
        self,
        document_id: str,
        claim_id: str,
    ) -> List[EvidenceBundle]:
        """
        Compatibility method required by the original agent interface.
        """

        return []