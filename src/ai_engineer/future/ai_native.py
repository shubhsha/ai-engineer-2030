"""AI-native development: pipeline arithmetic, spec checks, review queue.

Four small tools for Chapter 68. ``amdahl_speedup`` and ``lead_time`` do
the by-hand arithmetic of a delivery pipeline. ``check_spec`` joins a
specification to the tests that prove it. ``simulate_review`` plays a
review queue with a seeded random generator.
"""

from __future__ import annotations

import heapq
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

WORK_HOURS_PER_DAY = 8.0


# region amdahl
def amdahl_speedup(
    fractions: Sequence[float], speedups: Sequence[float]
) -> float:
    """Overall speedup when stage i (share f_i of the time) gets s_i."""
    if len(fractions) != len(speedups) or not fractions:
        raise ValueError("fractions and speedups must match and be non-empty")
    if any(f < 0 for f in fractions) or any(s <= 0 for s in speedups):
        raise ValueError("fractions must be >= 0 and speedups > 0")
    if not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
        raise ValueError("fractions must sum to 1")
    return 1.0 / sum(f / s for f, s in zip(fractions, speedups, strict=True))


# endregion


@dataclass(frozen=True, slots=True)
class Stage:
    """One stage of a delivery pipeline, with its duration in days."""

    name: str
    days: float

    def scaled(self, factor: float) -> Stage:
        """The same stage made ``factor`` times faster."""
        if factor <= 0:
            raise ValueError("factor must be positive")
        return Stage(self.name, self.days / factor)


def lead_time(stages: Sequence[Stage]) -> float:
    """Total days from idea to release when stages run in sequence."""
    return sum(s.days for s in stages)


# region spec
@dataclass(frozen=True, slots=True)
class Criterion:
    """One acceptance criterion and the test that proves it."""

    id: str
    text: str
    test: str


@dataclass(frozen=True, slots=True)
class Spec:
    """A named specification: a tuple of acceptance criteria."""

    name: str
    criteria: tuple[Criterion, ...]


@dataclass(frozen=True, slots=True)
class SpecCheck:
    """Criteria without a test, and tests without a criterion."""

    missing: tuple[str, ...]
    orphans: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing and not self.orphans


def check_spec(spec: Spec, test_names: Sequence[str]) -> SpecCheck:
    """Fail-closed join of a spec to a suite's test names."""
    names = set(test_names)
    claimed = {c.test for c in spec.criteria}
    missing = tuple(c.id for c in spec.criteria if c.test not in names)
    orphans = tuple(sorted(n for n in names if n not in claimed))
    return SpecCheck(missing, orphans)


# endregion


@dataclass(frozen=True, slots=True)
class ReviewResult:
    """Outcome of one review-queue run. Times are working hours."""

    submitted: int
    reviewed: int
    queue_length: int
    p50_hours: float
    p90_hours: float
    p99_hours: float


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    k = min(len(sorted_values) - 1, math.ceil(q * len(sorted_values)) - 1)
    return sorted_values[max(k, 0)]


# region review_queue
def simulate_review(
    arrivals_per_day: float,
    reviewers: int,
    review_hours: float,
    days: int,
    rng: random.Random,
) -> ReviewResult:
    """First-come first-served review queue in working hours.

    Pull requests arrive as a Poisson process; each review takes an
    exponential time with mean ``review_hours``. ``queue_length`` counts
    requests still unfinished when the last day ends.
    """
    if min(arrivals_per_day, review_hours) <= 0 or min(reviewers, days) < 1:
        raise ValueError("all inputs must be positive")
    horizon = days * WORK_HOURS_PER_DAY
    rate = arrivals_per_day / WORK_HOURS_PER_DAY
    free = [0.0] * reviewers
    heapq.heapify(free)
    leads: list[float] = []
    submitted = unfinished = 0
    t = rng.expovariate(rate)
    while t < horizon:
        submitted += 1
        start = max(t, heapq.heappop(free))
        done = start + rng.expovariate(1.0 / review_hours)
        heapq.heappush(free, done)
        if done <= horizon:
            leads.append(done - t)
        else:
            unfinished += 1
        t += rng.expovariate(rate)
    leads.sort()
    return ReviewResult(
        submitted,
        len(leads),
        unfinished,
        _percentile(leads, 0.5),
        _percentile(leads, 0.9),
        _percentile(leads, 0.99),
    )


# endregion
