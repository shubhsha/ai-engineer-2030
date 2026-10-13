"""Chapter 48: service level objectives, burn rates and input drift.

An SLO turns "is it healthy?" into a number you can spend. The error
budget is the bad minutes (or bad requests) the target allows, and the
burn rate says how fast the current errors are spending it.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Slo:
    name: str
    target: float  # share of good events, for example 0.995

    def __post_init__(self) -> None:
        if not 0.0 < self.target < 1.0:
            raise ValueError("target must be strictly between 0 and 1")

    @property
    def allowed_bad(self) -> float:
        return 1.0 - self.target


# region budget_math
def error_budget(slo: Slo, window: float) -> float:
    """Bad units the SLO allows in ``window`` (minutes, or requests)."""
    return slo.allowed_bad * window


def burn_rate(errors: float, total: float, slo: Slo) -> float:
    """1.0 spends the budget exactly over the SLO window; 14.4 spends
    2 percent of a 30-day budget in one hour."""
    if total <= 0:
        return 0.0
    return (errors / total) / slo.allowed_bad


# endregion budget_math


@dataclass(frozen=True, slots=True)
class Bucket:
    """Requests and errors counted in one interval that starts at ``t``."""

    t: float
    total: int
    errors: int


def _when(bucket: Bucket) -> float:
    return bucket.t


@dataclass(frozen=True, slots=True)
class Rule:
    """Fire when BOTH windows burn at ``threshold`` or faster."""

    name: str
    long_s: float
    short_s: float
    threshold: float
    severity: str  # "page" or "ticket"


# The SRE Workbook's recommended starting points for a 30-day SLO.
DEFAULT_RULES = (
    Rule("fast-1h", 3600.0, 300.0, 14.4, "page"),
    Rule("slow-6h", 21600.0, 1800.0, 6.0, "page"),
    Rule("leak-3d", 259200.0, 21600.0, 1.0, "ticket"),
)


# region multiwindow
@dataclass(frozen=True, slots=True)
class MultiWindowAlert:
    slo: Slo
    rules: tuple[Rule, ...] = DEFAULT_RULES

    def window_burn(
        self, events: Sequence[Bucket], now: float, span: float
    ) -> float:
        """Burn rate over the ``span`` seconds that end at ``now``.
        ``events`` must be sorted by time."""
        lo = bisect_right(events, now - span, key=_when)
        hi = bisect_right(events, now, key=_when)
        part = events[lo:hi]
        return burn_rate(
            sum(e.errors for e in part), sum(e.total for e in part), self.slo
        )

    def evaluate(self, events: Sequence[Bucket], now: float) -> list[Rule]:
        """The rules that fire at ``now``. The short window makes an
        alert stop as soon as the problem does."""
        fired = []
        for rule in self.rules:
            long = self.window_burn(events, now, rule.long_s)
            short = self.window_burn(events, now, rule.short_s)
            if long >= rule.threshold and short >= rule.threshold:
                fired.append(rule)
        return fired


# endregion multiwindow


# region psi_drift
def psi(
    expected: Sequence[float], actual: Sequence[float], bins: int
) -> float:
    """Population stability index of ``actual`` against ``expected``.

    Bin edges are quantiles of ``expected``, so each bin starts with
    about ``1 / bins`` of the baseline. An empty bin counts as 0.0001.
    """
    if bins < 2 or not expected or not actual:
        raise ValueError("need two bins and two non-empty samples")
    ordered = sorted(expected)
    edges = [
        ordered[min(len(ordered) - 1, i * len(ordered) // bins)]
        for i in range(1, bins)
    ]

    def shares(values: Sequence[float]) -> list[float]:
        counts = [0] * bins
        for v in values:
            counts[bisect_left(edges, v)] += 1
        return [max(c / len(values), 1e-4) for c in counts]

    return psi_shares(shares(expected), shares(actual))


def psi_shares(expected: Sequence[float], actual: Sequence[float]) -> float:
    """Sum of (a - e) * ln(a / e) over the bins."""
    return sum(
        (a - e) * math.log(a / e)
        for e, a in zip(expected, actual, strict=True)
    )


# endregion psi_drift
