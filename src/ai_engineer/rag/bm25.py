"""Chapter 29: BM25, the lexical ranking that hybrid search still needs.

Embeddings match meaning; BM25 matches the exact words, codes and names
that embeddings blur. Okapi BM25 as described by Robertson and Zaragoza
(2009), with the non-negative IDF that Lucene uses.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

_WORD = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase words, keeping codes such as ``e-4127`` whole."""
    return _WORD.findall(text.lower())


class BM25:
    def __init__(
        self, docs: Sequence[str], *, k1: float = 1.2, b: float = 0.75
    ) -> None:
        self._tf = [Counter(tokenize(doc)) for doc in docs]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = (sum(self._len) / len(docs)) if docs else 1.0
        df = Counter(term for tf in self._tf for term in tf)
        n = len(docs)
        self._idf = {
            term: math.log(1 + (n - f + 0.5) / (f + 0.5))
            for term, f in df.items()
        }
        self.k1, self.b = k1, b

    def score(self, query: str, i: int) -> float:
        """Rare query terms count most; repeats saturate (``k1``); long
        documents are discounted (``b``)."""
        tf = self._tf[i]
        norm = self.k1 * (1 - self.b + self.b * self._len[i] / self._avg)
        return sum(
            self._idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm)
            for t in set(tokenize(query))
            if t in tf
        )

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """The ``k`` best documents with a positive score, best first."""
        scored = [(i, self.score(query, i)) for i in range(len(self._tf))]
        hits = [(i, s) for i, s in scored if s > 0]
        return sorted(hits, key=lambda hit: (-hit[1], hit[0]))[:k]
