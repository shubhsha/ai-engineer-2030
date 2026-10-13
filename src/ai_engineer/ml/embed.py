"""Chapter 20: a deterministic embedding for offline tests and demos.

Real embeddings come from a trained model and place texts with similar
meaning near each other. This one hashes character trigrams into a
fixed-size vector, so it captures spelling, not meaning. It needs no
model, no network and no randomness, which makes it the right stand-in
when Volume III tests retrieval code offline.
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection

import numpy as np
from numpy.typing import NDArray

from ai_engineer.ml.numerics import Array, IntArray, normalize

Int8Array = NDArray[np.int8]


def hashed_embedding(text: str, dim: int = 256, n: int = 3) -> Array:
    """Embed ``text`` as a unit vector of signed, hashed character n-grams.

    Python's built-in ``hash`` is salted per process, so the same text
    would land in different buckets on every run; a cryptographic hash
    is stable everywhere.
    """
    vec = np.zeros(dim)
    padded = f" {' '.join(text.lower().split())} "
    for i in range(len(padded) - n + 1):
        digest = hashlib.blake2b(padded[i : i + n].encode(), digest_size=8)
        h = int.from_bytes(digest.digest(), "little")
        vec[h % dim] += 1.0 if h >> 63 else -1.0
    return normalize(vec)


def embed_all(texts: list[str], dim: int = 256) -> Array:
    return np.stack([hashed_embedding(t, dim) for t in texts])


def top_k(query: Array, matrix: Array, k: int) -> IntArray:
    """Rows of ``matrix`` most similar to ``query``, best first.

    Both sides are unit vectors, so the dot product is the cosine.
    """
    scores = matrix @ query
    k = min(k, len(scores))
    best = np.argpartition(-scores, k - 1)[:k]
    return np.asarray(best[np.argsort(-scores[best])], dtype=np.int64)


def recall_at_k(found: list[int], relevant: Collection[int], k: int) -> float:
    """Share of the relevant rows that appear among the first ``k`` found."""
    if not relevant:
        raise ValueError("no relevant rows to recall")
    return len(set(found[:k]) & set(relevant)) / len(relevant)


def quantize_int8(vectors: Array) -> tuple[Int8Array, Array]:
    """Keep one byte per number plus one scale per row: max entry is 127."""
    scale: Array = np.abs(vectors).max(axis=1, keepdims=True) / 127.0
    scale[scale == 0] = 1.0  # an all-zero row stays zero
    return np.round(vectors / scale).astype(np.int8), scale


def dequantize_int8(codes: Int8Array, scale: Array) -> Array:
    return np.asarray(codes * scale, dtype=np.float64)
