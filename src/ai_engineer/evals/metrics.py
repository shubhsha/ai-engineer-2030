"""Chapter 38: the statistics an evaluation needs, and no more."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


def wilson_interval(
    successes: int, n: int, z: float = 1.96
) -> tuple[float, float]:
    """Confidence interval for a pass rate (95% by default).

    Unlike the textbook p +/- z * sqrt(p(1 - p) / n), it stays inside
    [0, 1] and does not collapse to zero width at 0% or 100%.
    """
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError(f"need 0 <= successes <= n and n > 0, got {n}")
    p = successes / n
    scale = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / scale
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / scale
    return max(0.0, centre - half), min(1.0, centre + half)


def _check(n: int, c: int, k: int) -> None:
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError(f"need 0 <= c <= n and 1 <= k <= n: {n=} {c=} {k=}")


def pass_at_k(n: int, c: int, k: int) -> float:
    """Chance that at least one of ``k`` attempts passes, estimated
    without bias from ``n`` attempts of which ``c`` passed."""
    _check(n, c, k)
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Chance that all ``k`` attempts pass (pass^k): the reliability an
    agent needs when every run has to succeed."""
    _check(n, c, k)
    return math.comb(c, k) / math.comb(n, k)


@dataclass(frozen=True, slots=True)
class Paired:
    only_a: int  # cases A passed and B failed
    only_b: int  # cases B passed and A failed
    p_value: float


def paired_compare(a: Sequence[bool], b: Sequence[bool]) -> Paired:
    """Compare two systems run on the same cases (exact McNemar test).

    Cases both pass or both fail say nothing about which is better, so
    only the discordant ones count. Under "no difference" each is a
    fair coin flip; the p-value is the two-sided binomial tail.
    """
    if len(a) != len(b):
        raise ValueError("both systems must run on the same cases")
    only_a = sum(x and not y for x, y in zip(a, b, strict=True))
    only_b = sum(y and not x for x, y in zip(a, b, strict=True))
    n = only_a + only_b
    if n == 0:
        return Paired(0, 0, 1.0)
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1))
    return Paired(only_a, only_b, min(1.0, 2 * tail / 2**n))
