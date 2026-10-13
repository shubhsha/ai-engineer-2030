"""Chapter 15: a logistic-regression baseline, leak-free splits, metrics.

The model is deliberately small. It is the baseline that every fancier
model in a project has to beat, and its training loop is the loop that
trains a transformer, minus the size.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ai_engineer.ml.numerics import Array, IntArray


def sigmoid(z: Array) -> Array:
    """Logistic function, written via tanh so it never overflows."""
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def log_loss(y: Array, p: Array, eps: float = 1e-12) -> float:
    """Binary cross-entropy of probabilities ``p`` against 0/1 labels."""
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


@dataclass
class LogisticRegression:
    """Binary classifier trained by full-batch gradient descent."""

    lr: float = 0.5
    l2: float = 1e-3
    epochs: int = 300
    w: Array = field(default_factory=lambda: np.zeros(0))
    b: float = 0.0

    def fit(self, x: Array, y: Array) -> list[float]:
        """Train on features ``x`` (n, d) and labels ``y`` in {0, 1}.

        Returns the training loss before each update, so a caller can
        see whether the model is still learning.
        """
        n, d = x.shape
        self.w, self.b = np.zeros(d), 0.0
        history = []
        for _ in range(self.epochs):
            p = self.predict_proba(x)
            history.append(log_loss(y, p))
            err = p - y  # gradient of the loss with respect to the logit
            self.w -= self.lr * (x.T @ err / n + self.l2 * self.w)
            self.b -= self.lr * float(np.mean(err))
        return history

    def predict_proba(self, x: Array) -> Array:
        return sigmoid(x @ self.w + self.b)

    def predict(self, x: Array, threshold: float = 0.5) -> IntArray:
        return (self.predict_proba(x) >= threshold).astype(np.int64)


def group_split(
    groups: Sequence[str], test_fraction: float, seed: int
) -> tuple[IntArray, IntArray]:
    """Split row indices so that every group lands on one side only.

    Rows from one customer, document or session are correlated. If some
    reach training and some reach test, the test score measures memory,
    not generalization. Returns (train_rows, test_rows).
    """
    names = sorted(set(groups))
    order = np.random.default_rng(seed).permutation(len(names))
    n_test = max(1, round(len(names) * test_fraction))
    held_out = {names[i] for i in order[:n_test]}
    is_test = np.array([g in held_out for g in groups])
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


@dataclass(frozen=True)
class Scores:
    precision: float
    recall: float
    f1: float
    accuracy: float


def classification_scores(y_true: IntArray, y_pred: IntArray) -> Scores:
    """Precision, recall, F1 and accuracy for the positive class 1."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return Scores(precision, recall, f1, float(np.mean(y_true == y_pred)))
