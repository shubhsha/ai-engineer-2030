"""Chapter 19: scaled dot-product attention, masks, heads, KV cache.

``MultiHeadAttention.forward`` processes a whole sequence at once, as in
training and in the prefill phase of inference. ``step`` processes one
new token against a cache of earlier keys and values, as in decoding.
The tests check that the two paths agree.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ai_engineer.ml.numerics import Array, softmax

Mask = NDArray[np.bool_]


def causal_mask(n: int) -> Mask:
    """True where query position i may look at key position j (j <= i)."""
    return np.tril(np.ones((n, n), dtype=bool))


def padding_mask(lengths: Sequence[int], n: int) -> Mask:
    """Shape (batch, 1, n): True where key position j is a real token."""
    real = np.arange(n)[None, :] < np.asarray(lengths)[:, None]
    return real[:, None, :]


def sliding_window_mask(n: int, window: int) -> Mask:
    """Causal mask that keeps only the last ``window`` keys of each query."""
    i, j = np.arange(n)[:, None], np.arange(n)[None, :]
    return (j <= i) & (i - j < window)


def attention(q: Array, k: Array, v: Array, mask: Mask | None = None) -> Array:
    """softmax(q @ k.T / sqrt(d)) @ v over the last two axes.

    Each query scores every key, the scores become weights that sum to
    one, and the output is the weighted average of the values. Dividing
    by sqrt(d) keeps the scores' spread independent of the head size.
    """
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1])
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    return softmax(scores) @ v


@dataclass
class KVCache:
    """Keys and values of every token so far: (heads, tokens, head_dim)."""

    k: Array
    v: Array

    @classmethod
    def empty(cls, heads: int, head_dim: int) -> KVCache:
        return cls(
            np.zeros((heads, 0, head_dim)), np.zeros((heads, 0, head_dim))
        )

    def append(self, k: Array, v: Array) -> None:
        self.k = np.concatenate([self.k, k], axis=1)
        self.v = np.concatenate([self.v, v], axis=1)

    def __len__(self) -> int:
        return int(self.k.shape[1])


class MultiHeadAttention:
    """Causal self-attention with ``heads`` independent heads."""

    def __init__(
        self, d_model: int, heads: int, rng: np.random.Generator
    ) -> None:
        if d_model % heads:
            raise ValueError(
                f"d_model={d_model} is not divisible by heads={heads}"
            )
        self.heads, self.head_dim = heads, d_model // heads
        std = 1.0 / np.sqrt(d_model)
        self.w_q, self.w_k, self.w_v, self.w_o = (
            rng.normal(0.0, std, (d_model, d_model)) for _ in range(4)
        )

    def _split(self, x: Array) -> Array:
        """(tokens, d_model) -> (heads, tokens, head_dim)."""
        return x.reshape(len(x), self.heads, self.head_dim).transpose(1, 0, 2)

    def _merge(self, x: Array) -> Array:
        """(heads, tokens, head_dim) -> (tokens, d_model)."""
        return x.transpose(1, 0, 2).reshape(x.shape[1], -1)

    def forward(self, x: Array) -> Array:
        """Attend over a whole sequence ``x`` of shape (tokens, d_model)."""
        q, k, v = (self._split(x @ w) for w in (self.w_q, self.w_k, self.w_v))
        out = attention(q, k, v, causal_mask(len(x)))
        return self._merge(out) @ self.w_o

    def step(self, x_t: Array, cache: KVCache) -> Array:
        """Attend from one new token ``x_t`` (d_model,) to the cache.

        Only the new token's query, key and value are computed; the
        earlier keys and values are read from the cache, which grows by
        one entry. No mask is needed: the cache holds only the past.
        """
        x = x_t[None, :]
        q, k, v = (self._split(x @ w) for w in (self.w_q, self.w_k, self.w_v))
        cache.append(k, v)
        out = attention(q, cache.k, cache.v)
        return (self._merge(out) @ self.w_o).squeeze(0)


def grouped_query_attention(
    q: Array, k: Array, v: Array, mask: Mask | None = None
) -> Array:
    """Attention where many query heads share fewer key/value heads.

    ``q`` is (heads, n, d); ``k`` and ``v`` are (groups, m, d) with
    ``groups`` dividing ``heads``. Query head h reads key/value head
    ``h // (heads // groups)``. One group is multi-query attention;
    ``groups == heads`` is ordinary multi-head attention.
    """
    heads, groups = q.shape[0], k.shape[0]
    if heads % groups:
        raise ValueError(f"{heads} heads do not split into {groups} groups")
    reps = heads // groups
    k, v = np.repeat(k, reps, axis=0), np.repeat(v, reps, axis=0)
    return attention(q, k, v, mask)


def blockwise_attention(
    q: Array, k: Array, v: Array, block: int, mask: Mask | None = None
) -> Array:
    """Exact attention that reads the keys ``block`` at a time.

    The full score matrix is never built. Per query it keeps a running
    maximum ``m``, a running sum ``total`` of exp(score - m) and a
    running weighted sum ``acc`` of the values. When a block raises the
    maximum, the old ``total`` and ``acc`` shrink by exp(old m - new m).
    """
    m = np.full((*q.shape[:-1], 1), -np.inf)
    total = np.zeros_like(m)
    acc = np.zeros((*q.shape[:-1], v.shape[-1]))
    for lo in range(0, k.shape[-2], block):
        keys = k[..., lo : lo + block, :]
        s = q @ np.swapaxes(keys, -1, -2) / np.sqrt(q.shape[-1])
        if mask is not None:
            s = np.where(mask[..., lo : lo + block], s, -np.inf)
        m_new = np.maximum(m, s.max(axis=-1, keepdims=True))
        base = np.where(np.isinf(m_new), 0.0, m_new)  # nothing visible yet
        p = np.exp(s - base)
        shrink = np.exp(m - base)
        total = total * shrink + p.sum(axis=-1, keepdims=True)
        acc = acc * shrink + p @ v[..., lo : lo + block, :]
        m = m_new
    return acc / total


def kv_cache_bytes(
    layers: int,
    kv_heads: int,
    head_dim: int,
    tokens: int,
    bytes_per_value: int = 2,
) -> int:
    """Cache size: 2 (keys and values) x layers x kv_heads x head_dim
    x bytes_per_value, for each of ``tokens`` tokens."""
    return 2 * layers * kv_heads * head_dim * bytes_per_value * tokens
