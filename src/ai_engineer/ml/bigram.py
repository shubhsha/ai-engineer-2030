"""Chapter 21: the smallest language model, trained by counting.

A bigram model predicts the next character from the current one. It is
trained, evaluated and sampled exactly like a large language model: the
loss is the mean negative log-likelihood of the next token, and
generation repeats predict, sample, append. Only the predictor is
smaller.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from ai_engineer.ml.numerics import Array
from ai_engineer.ml.sampling import sample


@dataclass(frozen=True)
class BigramModel:
    chars: str
    counts: Array  # counts[i, j]: how often char j followed char i

    @classmethod
    def fit(cls, text: str, smoothing: float = 0.1) -> BigramModel:
        """Count transitions; ``smoothing`` keeps unseen pairs possible."""
        chars = "".join(sorted(set(text)))
        index = {c: i for i, c in enumerate(chars)}
        counts = np.full((len(chars), len(chars)), smoothing)
        for a, b in itertools.pairwise(text):
            counts[index[a], index[b]] += 1
        return cls(chars, counts)

    def logits(self, prev: str) -> Array:
        """Log-probabilities of every next character after ``prev``."""
        row = self.counts[self.chars.index(prev)]
        probs: Array = row / row.sum()
        return np.log(probs)

    def nll(self, text: str) -> float:
        """Mean negative log-likelihood per character of ``text``."""
        pairs = list(itertools.pairwise(text))
        total = sum(-self.logits(a)[self.chars.index(b)] for a, b in pairs)
        return float(total / len(pairs))

    def generate(
        self,
        start: str,
        length: int,
        rng: np.random.Generator,
        temperature: float = 1.0,
    ) -> str:
        out = [start]
        for _ in range(length):
            nxt = sample(self.logits(out[-1]), rng, temperature=temperature)
            out.append(self.chars[nxt])
        return "".join(out)
