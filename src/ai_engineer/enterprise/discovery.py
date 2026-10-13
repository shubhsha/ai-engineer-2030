"""Chapter 53: from discovery to handoff.

Four small checks that a person runs before they build and before they
leave: a problem statement that names a number, a readiness verdict that
can say stop, a use-case ranking whose sensitivity is measured, and a
handoff list that finds what nobody owns.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

Verdict3 = Literal["go", "fix", "stop"]


# region problem_statement
@dataclass(frozen=True, slots=True)
class ProblemStatement:
    who: str
    problem: str
    metric: str
    baseline: float
    target: float
    days: int

    def __post_init__(self) -> None:
        for name in ("who", "problem", "metric"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        if self.baseline == self.target:
            raise ValueError("target must differ from baseline")
        if self.days <= 0:
            raise ValueError("days must be positive")

    def text(self) -> str:
        return (
            f"{self.who} {self.problem}. We will know it is solved when "
            f"{self.metric} moves from {self.baseline:g} to "
            f"{self.target:g} within {self.days} days."
        )


# endregion problem_statement


@dataclass(frozen=True, slots=True)
class Threshold:
    go: float  # at or above: fine
    stop: float  # below: do not start

    def __post_init__(self) -> None:
        if not 0.0 <= self.stop <= self.go <= 1.0:
            raise ValueError("need 0 <= stop <= go <= 1")


DIMENSIONS = (
    "data_access",
    "data_quality",
    "labeled_examples",
    "owner",
    "risk_tier",
)
DEFAULT_THRESHOLDS: dict[str, Threshold] = {
    "data_access": Threshold(0.8, 0.4),
    "data_quality": Threshold(0.7, 0.4),
    "labeled_examples": Threshold(0.6, 0.2),
    "owner": Threshold(1.0, 0.5),
    "risk_tier": Threshold(0.7, 0.3),
}


@dataclass(frozen=True, slots=True)
class Readiness:
    verdict: Verdict3
    reasons: tuple[str, ...]


# region readiness
def readiness(
    scores: Mapping[str, float],
    thresholds: Mapping[str, Threshold] = DEFAULT_THRESHOLDS,
) -> Readiness:
    """One score per dimension, 0 to 1. One stop is a stop."""
    reasons: list[str] = []
    verdict: Verdict3 = "go"
    for dim, limit in thresholds.items():
        if dim not in scores:
            reasons.append(f"{dim}: not assessed")
            verdict = "stop"
            continue
        score = scores[dim]
        if score < limit.stop:
            reasons.append(f"{dim}: {score:g} below stop {limit.stop:g}")
            verdict = "stop"
        elif score < limit.go:
            reasons.append(f"{dim}: {score:g} below go {limit.go:g}")
            if verdict == "go":
                verdict = "fix"
    return Readiness(verdict, tuple(reasons))


# endregion readiness


@dataclass(frozen=True, slots=True)
class UseCase:
    name: str
    value: float  # 0 to 10
    feasibility: float  # 0 to 10
    risk: float  # 0 to 10, higher is worse


@dataclass(frozen=True, slots=True)
class Weights:
    value: float
    feasibility: float
    risk: float

    def __post_init__(self) -> None:
        if min(self.value, self.feasibility, self.risk) < 0:
            raise ValueError("weights must not be negative")
        if self.value + self.feasibility + self.risk <= 0:
            raise ValueError("weights must not all be zero")


# region ranking
def score(case: UseCase, w: Weights) -> float:
    total = w.value + w.feasibility + w.risk
    raw = (
        w.value * case.value
        + w.feasibility * case.feasibility
        + w.risk * (10.0 - case.risk)
    )
    return raw / total


def rank_use_cases(cases: Sequence[UseCase], weights: Weights) -> list[str]:
    """Names, best first; ties break by name so the order is stable."""
    ordered = sorted(cases, key=lambda c: (-score(c, weights), c.name))
    return [c.name for c in ordered]


@dataclass(frozen=True, slots=True)
class Stability:
    rankings: tuple[tuple[str, ...], ...]
    mean_tau: float  # 1.0 means every weight set gives one order
    top_share: float  # share of weight sets that agree on first place
    top3_overlap: float  # shared names in the top three, 0 to 1


def kendall_tau(a: Sequence[str], b: Sequence[str]) -> float:
    pos = {name: i for i, name in enumerate(b)}
    pairs = list(combinations(a, 2))
    agree = sum(1 for x, y in pairs if pos[x] < pos[y])
    return (2 * agree - len(pairs)) / len(pairs)


def rank_stability(
    cases: Sequence[UseCase], weight_sets: Sequence[Weights]
) -> Stability:
    ranks = tuple(tuple(rank_use_cases(cases, w)) for w in weight_sets)
    pairs = list(combinations(ranks, 2))
    tau = sum(kendall_tau(a, b) for a, b in pairs) / len(pairs)
    firsts = [r[0] for r in ranks]
    top = max(firsts.count(f) for f in set(firsts)) / len(ranks)
    sets = [set(r[:3]) for r in ranks]
    overlap = sum(len(a & b) / 3 for a, b in combinations(sets, 2)) / len(
        pairs
    )
    return Stability(ranks, tau, top, overlap)


# endregion ranking


@dataclass(frozen=True, slots=True)
class Component:
    name: str
    owner: str | None = None
    runbook: str | None = None
    slo: str | None = None
    evidence: str | None = None
    budget: str | None = None


NEEDS = ("owner", "runbook", "slo", "evidence", "budget")


# region handoff
def handoff_gaps(
    components: Sequence[Component],
) -> dict[str, tuple[str, ...]]:
    """Component name to the things it still lacks; clean ones are absent."""
    gaps: dict[str, tuple[str, ...]] = {}
    for comp in components:
        missing = tuple(
            need for need in NEEDS if not (getattr(comp, need) or "").strip()
        )
        if missing:
            gaps[comp.name] = missing
    return gaps


# endregion handoff
