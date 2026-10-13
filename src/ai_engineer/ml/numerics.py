"""Chapter 14: numerically stable building blocks for the ml package.

Softmax and log-sum-exp subtract the maximum before exponentiating, so
large scores cannot overflow. ``numerical_gradient`` is the slow, obvious
referee that every hand-written backward pass in this package is tested
against.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]


def logsumexp(x: Array, axis: int = -1) -> Array:
    """Compute ``log(sum(exp(x)))`` along ``axis`` without overflow."""
    m: Array = np.max(x, axis=axis, keepdims=True)
    s: Array = np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return np.squeeze(m + s, axis=axis)


def softmax(x: Array, axis: int = -1) -> Array:
    """Turn scores into probabilities that sum to one along ``axis``.

    Subtracting the maximum changes nothing mathematically, because
    softmax ignores a constant shift, but it keeps ``exp`` finite.
    """
    e: Array = np.exp(x - np.max(x, axis=axis, keepdims=True))
    total: Array = np.sum(e, axis=axis, keepdims=True)
    return e / total


def log_softmax(x: Array, axis: int = -1) -> Array:
    """Log-probabilities, computed without forming tiny probabilities."""
    return x - np.expand_dims(logsumexp(x, axis), axis)


def cross_entropy(logits: Array, targets: IntArray) -> float:
    """Mean negative log-probability of the correct class.

    ``logits`` has shape (n, classes); ``targets`` holds n class indices.
    This is the loss that trains classifiers and language models alike.
    """
    logp = log_softmax(logits)
    return float(-np.mean(logp[np.arange(len(targets)), targets]))


def numerical_gradient(
    f: Callable[[Array], float], x: Array, eps: float = 1e-6
) -> Array:
    """Estimate the gradient of ``f`` at ``x`` by central differences.

    Each element of ``x`` is nudged in place and restored, so ``f`` must
    read ``x`` rather than a copy. Two calls per element: far too slow
    for training, exactly right for checking a backward pass.
    """
    grad = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        saved = x[i]
        x[i] = saved + eps
        up = f(x)
        x[i] = saved - eps
        down = f(x)
        x[i] = saved
        grad[i] = (up - down) / (2 * eps)
    return grad


def relative_error(a: Array, b: Array) -> float:
    """Distance between two arrays, scaled by their sizes."""
    num = np.linalg.norm(a - b)
    den = max(float(np.linalg.norm(a) + np.linalg.norm(b)), 1e-12)
    return float(num / den)


def normalize(x: Array) -> Array:
    """Scale each row (the last axis) to length one; zero rows stay zero."""
    norm: Array = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def cosine_similarity(a: Array, b: Array) -> Array:
    """Cosine of the angle between each row of ``a`` and each row of ``b``."""
    return normalize(a) @ np.swapaxes(normalize(b), -1, -2)
