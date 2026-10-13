"""Chapter 16: a multilayer perceptron with hand-written backpropagation.

Every gradient here comes from the chain rule and is checked against
``numerical_gradient`` in the tests. Frameworks compute the same numbers
automatically; deriving them once is how you learn what they do.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ai_engineer.ml.numerics import Array, IntArray, cross_entropy, softmax

Params = dict[str, Array]


def init_mlp(sizes: Sequence[int], rng: np.random.Generator) -> Params:
    """He initialization: weights from N(0, 2 / fan_in), biases zero.

    The variance keeps activations from shrinking or exploding as they
    pass through many ReLU layers.
    """
    params: Params = {}
    for i, (fan_in, fan_out) in enumerate(itertools.pairwise(sizes)):
        std = np.sqrt(2.0 / fan_in)
        params[f"W{i}"] = rng.normal(0.0, std, (fan_in, fan_out))
        params[f"b{i}"] = np.zeros(fan_out)
    return params


def forward(params: Params, x: Array) -> tuple[Array, list[Array]]:
    """Return the logits and the input that each layer received."""
    layers = len(params) // 2
    inputs = []
    h = x
    for i in range(layers):
        inputs.append(h)
        z = h @ params[f"W{i}"] + params[f"b{i}"]
        h = np.maximum(z, 0.0) if i < layers - 1 else z
    return h, inputs


def loss_and_grads(
    params: Params, x: Array, y: IntArray
) -> tuple[float, Params]:
    """Softmax cross-entropy and its gradient for every parameter."""
    logits, inputs = forward(params, x)
    n = len(y)
    delta = softmax(logits)
    delta[np.arange(n), y] -= 1.0
    delta /= n  # now dLoss/dlogits
    grads: Params = {}
    for i in reversed(range(len(inputs))):
        grads[f"W{i}"] = inputs[i].T @ delta
        grads[f"b{i}"] = delta.sum(axis=0)
        if i:  # through the weights, then through the ReLU
            delta = (delta @ params[f"W{i}"].T) * (inputs[i] > 0)
    return cross_entropy(logits, y), grads


@dataclass
class Adam:
    """Adam: momentum plus a per-parameter step size (Kingma and Ba)."""

    lr: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    t: int = 0
    m: Params = field(default_factory=dict)
    v: Params = field(default_factory=dict)

    def step(self, params: Params, grads: Params) -> None:
        """Update ``params`` in place."""
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        for k, g in grads.items():
            m = self.m.setdefault(k, np.zeros_like(g))
            v = self.v.setdefault(k, np.zeros_like(g))
            m[...] = b1 * m + (1 - b1) * g  # running mean of gradients
            v[...] = b2 * v + (1 - b2) * g * g  # ... and of their squares
            m_hat = m / (1 - b1**self.t)  # undo the bias of starting at 0
            v_hat = v / (1 - b2**self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def train(
    params: Params,
    x: Array,
    y: IntArray,
    *,
    epochs: int,
    batch_size: int,
    opt: Adam,
    rng: np.random.Generator,
) -> list[float]:
    """Mini-batch training; returns the full-data loss after each epoch."""
    history = []
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(y), batch_size):
            rows = order[start : start + batch_size]
            _, grads = loss_and_grads(params, x[rows], y[rows])
            opt.step(params, grads)
        history.append(cross_entropy(forward(params, x)[0], y))
    return history


def predict(params: Params, x: Array) -> IntArray:
    logits, _ = forward(params, x)
    return np.asarray(logits.argmax(axis=-1), dtype=np.int64)
