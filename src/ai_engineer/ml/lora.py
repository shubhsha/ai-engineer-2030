"""Chapter 22: LoRA, a low-rank update trained beside a frozen weight.

A dense layer computes ``x @ W``. LoRA keeps ``W`` frozen and learns two
thin matrices, ``A`` (d_in x r) and ``B`` (r x d_out), adding
``scale * x @ A @ B``. With r much smaller than d, the update has a
tiny fraction of the parameters, and after training it can be merged
into ``W`` so inference costs nothing extra.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ai_engineer.ml.numerics import Array


@dataclass
class LoRALinear:
    weight: Array  # (d_in, d_out), frozen
    a: Array  # (d_in, r), random at the start
    b: Array  # (r, d_out), zero at the start, so the update starts at 0
    alpha: float

    @classmethod
    def wrap(
        cls, weight: Array, rank: int, alpha: float, rng: np.random.Generator
    ) -> LoRALinear:
        d_in, d_out = weight.shape
        a = rng.normal(0.0, d_in**-0.5, (d_in, rank))
        return cls(weight, a, np.zeros((rank, d_out)), alpha)

    @property
    def rank(self) -> int:
        return int(self.a.shape[1])

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    def __call__(self, x: Array) -> Array:
        return x @ self.weight + self.scale * (x @ self.a) @ self.b

    def grads(self, x: Array, grad_out: Array) -> tuple[Array, Array]:
        """Gradients for ``a`` and ``b`` given dLoss/dOutput.

        ``weight`` gets no gradient and no optimizer state: that is where
        the memory saving of parameter-efficient fine-tuning comes from.
        """
        grad_b = self.scale * (x @ self.a).T @ grad_out
        grad_a = self.scale * x.T @ (grad_out @ self.b.T)
        return grad_a, grad_b

    def merged(self) -> Array:
        """``W + scale * A @ B``: one matrix, no extra inference cost."""
        return self.weight + self.scale * self.a @ self.b

    @property
    def trainable(self) -> int:
        return self.a.size + self.b.size


def fit(
    layer: LoRALinear, x: Array, y: Array, *, steps: int, lr: float
) -> list[float]:
    """Gradient descent on mean squared error; returns the loss curve."""
    losses = []
    for _ in range(steps):
        err = layer(x) - y
        losses.append(float(np.mean(err**2)))
        grad_a, grad_b = layer.grads(x, 2 * err / err.size)
        layer.a -= lr * grad_a
        layer.b -= lr * grad_b
    return losses
