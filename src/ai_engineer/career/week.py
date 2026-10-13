"""A weekly planner: capacity, a buffer, ranked commitments and a review."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Commitment:
    """One outcome you promise for the week."""

    title: str
    impact: float
    hours: float

    @property
    def ratio(self) -> float:
        return self.impact / self.hours


# region week_model
@dataclass(frozen=True, slots=True)
class Week:
    """Hours in the week, hours in meetings, hours held back for the
    interruptions you cannot schedule."""

    capacity: float
    meetings: float
    buffer: float

    @property
    def available(self) -> float:
        return max(0.0, self.capacity - self.meetings - self.buffer)


@dataclass(frozen=True, slots=True)
class Plan:
    chosen: tuple[Commitment, ...]
    dropped: tuple[Commitment, ...]
    free: float

    @property
    def planned_hours(self) -> float:
        return sum(c.hours for c in self.chosen)


def plan(week: Week, commitments: Sequence[Commitment]) -> Plan:
    """Take commitments by impact per hour, while they fit."""
    left = week.available
    chosen: list[Commitment] = []
    dropped: list[Commitment] = []
    for c in sorted(commitments, key=lambda c: (-c.ratio, c.title)):
        if c.hours <= left:
            chosen.append(c)
            left -= c.hours
        else:
            dropped.append(c)
    return Plan(tuple(chosen), tuple(dropped), left)


# endregion


# region fragmentation
def fragmentation(blocks: Sequence[float], min_minutes: float = 90) -> int:
    """Count the free blocks (in minutes) long enough for deep work."""
    return sum(1 for b in blocks if b >= min_minutes)


# endregion


# region review
@dataclass(frozen=True, slots=True)
class Review:
    ratio: float
    next_buffer: float


def review(
    planned: Sequence[float],
    done: Sequence[float],
    interruptions: float,
    buffer: float,
    smoothing: float = 0.5,
) -> Review:
    """Compare planned with actual hours and move the buffer toward what
    the interruptions really cost. `done[i]` is the time item i took."""
    total = sum(planned)
    ratio = sum(done) / total if total else 1.0
    nxt = (1 - smoothing) * buffer + smoothing * interruptions
    return Review(ratio, nxt)


# endregion


# region simulate
def simulate(
    rng: random.Random, weeks: int = 12, correct: bool = True
) -> list[float]:
    """Overcommitment per week: hours needed beyond hours available.

    Estimates run low by a random 1.2x to 1.9x; interruptions run 3 to 9
    hours. With `correct`, each week's estimates are scaled by the mean
    ratio of the earlier reviews and the buffer follows `review`.
    """
    pool = [Commitment(f"t{i}", 1 + i % 5, 3 + i % 4) for i in range(8)]
    buffer = 2.0
    ratios: list[float] = []
    over: list[float] = []
    for _ in range(weeks):
        scale = sum(ratios) / len(ratios) if correct and ratios else 1.0
        week = Week(40, 14, buffer)
        view = [Commitment(c.title, c.impact, c.hours * scale) for c in pool]
        chosen = plan(week, view).chosen
        names = {c.title for c in chosen}
        est = [c.hours for c in pool if c.title in names]
        actual = [h * rng.uniform(1.2, 1.9) for h in est]
        hit = rng.uniform(3, 9)
        r = review(est, actual, hit, buffer)
        over.append(max(0.0, sum(actual) + hit - (40 - 14)))
        ratios.append(r.ratio)
        if correct:
            buffer = r.next_buffer
    return over


# endregion
