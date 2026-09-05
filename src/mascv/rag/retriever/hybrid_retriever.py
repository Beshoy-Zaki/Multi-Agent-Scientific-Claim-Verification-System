"""Hybrid dense-sparse retriever combining BM25 and lexical similarity.

The full production design (see docs/rag_and_evidence) calls for combining a
sparse keyword signal (BM25) with a dense embedding similarity signal. A real
embedding model (e.g. sentence-transformers) is intentionally kept pluggable
behind ``BaseEmbeddingModel`` so it can be swapped in without touching this
class. Until a concrete embedding model is wired in, the "dense" side of the
hybrid score falls back to a lightweight bag-of-words cosine similarity so the
retriever is fully functional (and testable offline) rather than a stub that
raises ``NotImplementedError``.
"""

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

from mascv.rag.embeddings.base_embedding import BaseEmbeddingModel

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    """Lowercase, alphanumeric tokenization shared by BM25 and the lexical fallback."""
    return _TOKEN_RE.findall(text.lower())


def _cosine_similarity(query_tokens: List[str], doc_tokens: List[str]) -> float:
    """Simple term-frequency cosine similarity used as a dense-score fallback."""

    if not query_tokens or not doc_tokens:
        return 0.0

    query_counts = Counter(query_tokens)
    doc_counts = Counter(doc_tokens)

    shared_terms = set(query_counts) & set(doc_counts)

    dot_product = sum(
        query_counts[term] * doc_counts[term] for term in shared_terms
    )

    query_norm = math.sqrt(sum(count**2 for count in query_counts.values()))
    doc_norm = math.sqrt(sum(count**2 for count in doc_counts.values()))

    if query_norm == 0 or doc_norm == 0:
        return 0.0

    return dot_product / (query_norm * doc_norm)


def _normalize(scores: List[float]) -> List[float]:
    """Min-max normalize a list of scores into [0, 1]."""

    if not scores:
        return scores

    lo, hi = min(scores), max(scores)

    if hi - lo < 1e-12:
        return [0.0 for _ in scores]

    return [(score - lo) / (hi - lo) for score in scores]


def _vector_cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two dense embedding vectors."""

    if not a or not b:
        return 0.0

    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


class HybridRetriever:
    """Retrieves candidate chunks using hybrid sparse (BM25) and dense/lexical similarity."""

    def __init__(
        self,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        sparse_weight: float = 0.5,
        dense_weight: float = 0.5,
    ) -> None:
        self.embedding_model = embedding_model
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight

    def retrieve(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 15,
    ) -> List[Dict[str, Any]]:
        """Execute hybrid search across the supplied candidate chunks.

        Each returned chunk is the original chunk dict augmented with a
        ``retrieval_score`` field (combined, normalized sparse + dense score)
        so downstream rerankers (see ``reranker.py``) can further adjust it.
        """

        if not chunks:
            return []

        query_tokens = _tokenize(query)
        tokenized_corpus = [_tokenize(chunk.get("text", "")) for chunk in chunks]

        # Sparse signal: BM25 over the candidate chunks.
        if any(tokenized_corpus):
            bm25 = BM25Okapi(tokenized_corpus)
            sparse_scores = list(bm25.get_scores(query_tokens))
        else:
            sparse_scores = [0.0 for _ in chunks]

        # Dense signal: real embeddings if provided, otherwise a lexical
        # cosine-similarity fallback that behaves reasonably without any
        # external model weights.
        if self.embedding_model is not None:
            query_vector = self.embedding_model.embed_query(query)
            doc_vectors = self.embedding_model.embed_texts(
                [chunk.get("text", "") for chunk in chunks]
            )
            dense_scores = [
                _vector_cosine(query_vector, doc_vector)
                for doc_vector in doc_vectors
            ]
        else:
            dense_scores = [
                _cosine_similarity(query_tokens, doc_tokens)
                for doc_tokens in tokenized_corpus
            ]

        norm_sparse = _normalize(sparse_scores)
        norm_dense = _normalize(dense_scores)

        scored_chunks = []

        for chunk, sparse_score, dense_score in zip(chunks, norm_sparse, norm_dense):
            combined = (
                self.sparse_weight * sparse_score
                + self.dense_weight * dense_score
            )

            scored_chunk = dict(chunk)
            scored_chunk["retrieval_score"] = combined
            scored_chunk["sparse_score"] = sparse_score
            scored_chunk["dense_score"] = dense_score

            scored_chunks.append(scored_chunk)

        scored_chunks.sort(key=lambda item: item["retrieval_score"], reverse=True)

        return scored_chunks[:top_k]
