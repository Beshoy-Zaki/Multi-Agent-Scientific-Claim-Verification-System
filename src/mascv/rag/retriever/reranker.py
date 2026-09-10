"""Evidence candidate reranking."""

from typing import List, Dict, Any


def rerank_chunks(
    chunks: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Rerank retrieved chunks using retrieval quality
    and scientific evidence indicators.
    """

    for chunk in chunks:
        score = chunk.get("retrieval_score", 0.0)

        text = chunk["text"].lower()

        # Results sections are usually more useful
        # for performance claims.
        section = chunk.get("section", "").lower()

        if "results" in section:
            score += 0.10

        if "experiment" in section:
            score += 0.05

        if "evaluation" in section:
            score += 0.05

        # Tables and numerical results are valuable.
        if "table" in text:
            score += 0.05

        if "%" in text:
            score += 0.03

        chunk["rerank_score"] = score

    return sorted(
        chunks,
        key=lambda x: x["rerank_score"],
        reverse=True,
    )[:top_k]


class EvidenceReranker:
    """Thin class wrapper around :func:`rerank_chunks`.

    ``evidence_rag.py`` uses the ``rerank_chunks`` function directly, but the
    RAG pipeline is documented (and tested, see ``tests/unit/test_rag``) as a
    class-based component alongside ``HybridRetriever``. This wrapper keeps
    both call styles working without duplicating the ranking logic.
    """

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k

    def rerank(
        self,
        chunks: List[Dict[str, Any]],
        top_k: int = None,
    ) -> List[Dict[str, Any]]:
        return rerank_chunks(chunks, top_k=top_k or self.top_k)