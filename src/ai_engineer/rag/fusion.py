"""Chapter 29: reciprocal rank fusion."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence


def rrf(
    rankings: Iterable[Sequence[str]], *, k: int = 60
) -> list[tuple[str, float]]:
    """Merge ranked lists by summing ``1 / (k + rank)`` for each item.

    Only ranks are combined, so BM25 scores and cosine similarities
    never have to be put on one scale. ``k = 60`` comes from Cormack,
    Clarke and Büttcher (2009); it damps the influence of top ranks.
    """
    scores: defaultdict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
