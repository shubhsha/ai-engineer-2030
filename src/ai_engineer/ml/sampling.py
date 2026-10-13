"""Chapter 21: from next-token scores to a choice.

A language model returns a score (logit) for every token in its
vocabulary. Decoding turns those scores into one token: greedily, or by
sampling after temperature scaling and top-k or top-p (nucleus)
filtering. The same model can be dull or erratic depending only on
these few lines.
"""

from __future__ import annotations

import math

import numpy as np

from ai_engineer.ml.numerics import Array, softmax


def top_k_filter(logits: Array, k: int) -> Array:
    """Keep the ``k`` highest scores (and any ties); drop the rest."""
    if k >= logits.size:
        return logits.copy()
    cutoff = np.partition(logits, -k)[-k]
    return np.where(logits >= cutoff, logits, -np.inf)


def top_p_filter(logits: Array, p: float) -> Array:
    """Keep the smallest set of top tokens whose probability reaches ``p``.

    A token survives if the tokens ranked above it hold less than ``p``
    of the mass, so the token that crosses the threshold is kept.
    """
    order = np.argsort(-logits, kind="stable")
    probs = softmax(logits[order])
    keep = order[np.cumsum(probs) - probs < p]
    out = np.full_like(logits, -np.inf)
    out[keep] = logits[keep]
    return out


def sample(
    logits: Array,
    rng: np.random.Generator,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> int:
    """Choose a token id. Temperature 0 means greedy (always the best)."""
    if temperature == 0:
        return int(np.argmax(logits))
    z = logits / temperature
    if top_k is not None:
        z = top_k_filter(z, top_k)
    if top_p is not None:
        z = top_p_filter(z, top_p)
    return int(rng.choice(len(z), p=softmax(z)))


def entropy_bits(logits: Array) -> float:
    """How uncertain the distribution is: 0 bits means one certain token."""
    p = softmax(logits)
    nz = p[p > 0]
    return float(-np.sum(nz * np.log2(nz)))


def perplexity(mean_nll: float) -> float:
    """exp(mean negative log-likelihood): the effective branching factor."""
    return math.exp(mean_nll)
