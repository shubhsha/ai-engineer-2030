"""Listening to a team: an anonymous survey, an interval, a delegation record.

A team survey has few people, so two rules matter more than any chart:
never report a group smaller than a floor, and report an interval, not a
point. Delegation is the other half of leading: a level and a date.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
from statistics import mean


# region survey
@dataclass(frozen=True, slots=True)
class SurveyResponse:
    """One person's answer: a score from 1 to 7 on one item."""

    item: str
    score: int
    quarter: str
    group: str

    def __post_init__(self) -> None:
        if not 1 <= self.score <= 7:
            raise ValueError(f"score must be 1 to 7, got {self.score}")


@dataclass(frozen=True, slots=True)
class GroupReport:
    """What may be shown for one (group, item, quarter) cell."""

    group: str
    item: str
    quarter: str
    n: int
    mean: float


def aggregate(
    responses: list[SurveyResponse], floor: int = 5
) -> list[GroupReport]:
    """Mean score per cell; cells with fewer than `floor` answers vanish."""
    cells: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for r in responses:
        cells[(r.group, r.item, r.quarter)].append(r.score)
    return [
        GroupReport(g, i, q, len(s), mean(s))
        for (g, i, q), s in sorted(cells.items())
        if len(s) >= floor
    ]


# endregion survey


# region bootstrap
def bootstrap_interval(
    scores: list[float],
    rng: random.Random,
    n: int = 2000,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean (Efron, 1979)."""
    if not scores:
        raise ValueError("no scores")
    means = sorted(mean(rng.choices(scores, k=len(scores))) for _ in range(n))
    tail = (1 - level) / 2
    return means[int(tail * n)], means[int((1 - tail) * n) - 1]


def change_is_clear(
    before: tuple[float, float], after: tuple[float, float]
) -> bool:
    """True when the two intervals do not overlap."""
    return after[0] > before[1] or after[1] < before[0]


# endregion bootstrap


# region delegation
class DelegationLevel(IntEnum):
    """How much the owner decides, from least (1) to most (6)."""

    TELL = 1  # I decide and tell you
    SELL = 2  # I decide and explain why
    CONSULT = 3  # I ask what you think, then decide
    AGREE = 4  # we decide together
    ADVISE = 5  # you decide after hearing my view
    INFORM = 6  # you decide, tell me if you want


@dataclass(frozen=True, slots=True)
class Delegation:
    """A task handed over: who, what level, when to look again."""

    task: str
    owner: str
    level: DelegationLevel
    check_in: str

    def decides(self, who: str) -> bool:
        """The owner decides only from ADVISE up; below that the lead does."""
        return who == self.owner and self.level >= DelegationLevel.ADVISE


# endregion delegation
