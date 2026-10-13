"""Chapter 29: hybrid retrieval, and the metrics that judge it."""

from __future__ import annotations

from collections.abc import Collection, Sequence

from ai_engineer.ml.embed import embed_all, hashed_embedding, top_k
from ai_engineer.rag.bm25 import BM25
from ai_engineer.rag.chunk import Chunk
from ai_engineer.rag.fusion import rrf


class HybridIndex:
    """BM25 and embeddings over the same chunks, fused by rank.

    The embeddings here are the hashed trigrams of Chapter 20, so tests
    run offline; swap in a real embedding model for meaning.
    """

    def __init__(self, chunks: Sequence[Chunk], *, dim: int = 256) -> None:
        self.chunks = list(chunks)
        texts = [chunk.indexed for chunk in self.chunks]
        self._bm25 = BM25(texts)
        self._vectors = embed_all(texts, dim)
        self._dim = dim

    def lexical(self, query: str, k: int) -> list[str]:
        return [self.chunks[i].id for i, _ in self._bm25.search(query, k)]

    def dense(self, query: str, k: int) -> list[str]:
        vector = hashed_embedding(query, self._dim)
        return [self.chunks[i].id for i in top_k(vector, self._vectors, k)]

    def search(self, query: str, k: int = 5, *, depth: int = 20) -> list[str]:
        """Fuse the top ``depth`` of each ranking; return ``k`` chunk ids."""
        fused = rrf([self.lexical(query, depth), self.dense(query, depth)])
        return [chunk_id for chunk_id, _ in fused[:k]]


def recall_at_k(
    retrieved: Sequence[str], relevant: Collection[str], k: int
) -> float:
    """The share of relevant items found in the top ``k``."""
    if not relevant:
        raise ValueError("recall is undefined without relevant items")
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)


def reciprocal_rank(
    retrieved: Sequence[str], relevant: Collection[str]
) -> float:
    """1 / rank of the first relevant item, or 0 if none was found."""
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0
