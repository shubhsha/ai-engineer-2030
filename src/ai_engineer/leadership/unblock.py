"""A blocker log: how long people wait, on whom, and what Little's law says.

A blocker is one person waiting on another. Times are in working hours
from an arbitrary zero, so no real clock is involved. The log answers
four questions: how long do blockers last, which are still open, who is
waited on most, and does the average count of open blockers match the
arrival rate times the average duration (Little's law).
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass


# region blocker
@dataclass(frozen=True, slots=True)
class Blocker:
    """One person waiting on another. `cleared_at` is None while open."""

    raised_at: float
    waiting: str
    waited_on: str
    cleared_at: float | None = None

    def __post_init__(self) -> None:
        if self.cleared_at is not None and self.cleared_at < self.raised_at:
            raise ValueError("a blocker cannot clear before it is raised")

    @property
    def is_open(self) -> bool:
        return self.cleared_at is None

    def waited(self, now: float) -> float:
        """Hours waited so far, or in total once cleared."""
        end = now if self.cleared_at is None else min(self.cleared_at, now)
        return max(0.0, end - self.raised_at)


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile, p in (0, 100]."""
    if not values:
        raise ValueError("no values")
    ranked = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ranked)))
    return ranked[rank - 1]


def time_to_unblock(log: list[Blocker]) -> dict[str, float]:
    """Median, 90th percentile and mean of cleared blockers, in hours."""
    done = [b.waited(math.inf) for b in log if not b.is_open]
    return {
        "p50": percentile(done, 50),
        "p90": percentile(done, 90),
        "mean": sum(done) / len(done),
    }


def aging(log: list[Blocker], now: float) -> list[tuple[Blocker, float]]:
    """Open blockers with their age, oldest first."""
    rows = [(b, b.waited(now)) for b in log if b.is_open]
    return sorted(rows, key=lambda row: -row[1])


# endregion blocker


# region wait-graph
def wait_graph(log: list[Blocker], now: float) -> list[tuple[str, float]]:
    """People ranked by the total hours others waited on them."""
    total: dict[str, float] = defaultdict(float)
    for b in log:
        total[b.waited_on] += b.waited(now)
    return sorted(total.items(), key=lambda kv: (-kv[1], kv[0]))


def wait_share(log: list[Blocker], now: float, person: str) -> float:
    """The fraction of all waiting that was waiting on `person`."""
    ranked = dict(wait_graph(log, now))
    everyone = sum(ranked.values())
    return ranked.get(person, 0.0) / everyone if everyone else 0.0


# endregion wait-graph


# region littles
@dataclass(frozen=True, slots=True)
class LittlesCheck:
    """Little's law for one window: L = arrival rate times mean time."""

    arrival_rate: float
    mean_time: float
    predicted_open: float
    observed_open: float


def littles_check(
    log: list[Blocker], start: float, end: float
) -> LittlesCheck:
    """Compare lambda * W with the time-average count of open blockers.

    Blockers raised inside [start, end) count as arrivals. A blocker still
    open at `end` is clipped there, so the window is closed off cleanly.
    """
    span = end - start
    mine = [b for b in log if start <= b.raised_at < end]
    if not mine or span <= 0:
        raise ValueError("need arrivals and a positive window")
    waits = [b.waited(end) for b in mine]
    rate = len(mine) / span
    mean_time = sum(waits) / len(waits)
    return LittlesCheck(rate, mean_time, rate * mean_time, sum(waits) / span)


# endregion littles


# region simulate
def simulate_quarter(
    rng: random.Random,
    people: list[str],
    weights: list[float],
    per_hour: float,
    median_hours: float,
    hours: float = 520.0,
) -> list[Blocker]:
    """Seeded blockers: Poisson arrivals, lognormal clearing times.

    `weights` says how likely each person is to be the one waited on.
    """
    log: list[Blocker] = []
    t = rng.expovariate(per_hour)
    while t < hours:
        asked = rng.choices(people, weights)[0]
        waiting = rng.choice([p for p in people if p != asked])
        took = rng.lognormvariate(math.log(median_hours), 0.9)
        log.append(Blocker(t, waiting, asked, t + took))
        t += rng.expovariate(per_hour)
    return log


# endregion simulate
