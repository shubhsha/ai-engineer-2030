"""A transition planner: experiments, a runway and a decision scored with
criteria that were fixed before any option was scored."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

MAX_SCORE = 5
REQUIRED_CRITERION = "reversibility"


# region plan_experiment
@dataclass(frozen=True, slots=True)
class Experiment:
    """A cheap test of a role: one question, a price and a deadline."""

    question: str
    cost: Decimal
    deadline: date

    def overdue(self, today: date) -> bool:
        return today > self.deadline


def total_cost(experiments: Sequence[Experiment]) -> Decimal:
    return sum((e.cost for e in experiments), Decimal(0))


def overdue(
    experiments: Sequence[Experiment], today: date
) -> tuple[Experiment, ...]:
    """Experiments whose deadline passed; unanswered, they are habits."""
    return tuple(e for e in experiments if e.overdue(today))


# endregion plan_experiment


# region plan_runway
def runway_months(
    savings: Decimal, monthly_costs: Decimal, monthly_income: Decimal
) -> Decimal | None:
    """Months until savings reach zero; None if income covers costs."""
    if savings < 0 or monthly_costs < 0 or monthly_income < 0:
        raise ValueError("savings, costs and income cannot be negative")
    burn = monthly_costs - monthly_income
    if burn <= 0:
        return None
    return (savings / burn).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)


# endregion plan_runway


@dataclass(frozen=True, slots=True)
class Criterion:
    name: str
    weight: Decimal


@dataclass(frozen=True, slots=True)
class Ranked:
    option: str
    total: Decimal


@dataclass(frozen=True, slots=True)
class Decision:
    ranked: tuple[Ranked, ...]
    margin: Decimal

    @property
    def winner(self) -> str:
        return self.ranked[0].option


class OrderError(Exception):
    """A score was entered before the criteria were fixed, or the
    criteria were changed after scoring began."""


# region plan_scorecard
class Scorecard:
    """Fix criteria first, then score; the order is enforced."""

    def __init__(self) -> None:
        self._criteria: tuple[Criterion, ...] = ()
        self._scores: dict[tuple[str, str], int] = {}

    def fix_criteria(self, criteria: Sequence[Criterion]) -> None:
        if self._scores:
            raise OrderError("criteria cannot change after scoring began")
        names = [c.name for c in criteria]
        if len(set(names)) != len(names):
            raise ValueError("duplicate criterion")
        if REQUIRED_CRITERION not in names:
            raise ValueError(f"a {REQUIRED_CRITERION!r} criterion is required")
        if any(c.weight <= 0 for c in criteria):
            raise ValueError("weights must be positive")
        self._criteria = tuple(criteria)

    def score(self, option: str, criterion: str, value: int) -> None:
        if not self._criteria:
            raise OrderError("fix the criteria before scoring")
        if criterion not in {c.name for c in self._criteria}:
            raise ValueError(f"unknown criterion {criterion!r}")
        if not 0 <= value <= MAX_SCORE:
            raise ValueError(f"score must be 0 to {MAX_SCORE}")
        self._scores[(option, criterion)] = value

    # endregion plan_scorecard

    # region plan_decide
    def decide(self) -> Decision:
        options = sorted({o for o, _ in self._scores})
        if not options:
            raise ValueError("no options scored")
        totals: list[Ranked] = []
        for option in options:
            total = Decimal(0)
            for c in self._criteria:
                if (option, c.name) not in self._scores:
                    raise ValueError(f"{option!r} lacks {c.name!r}")
                total += c.weight * self._scores[(option, c.name)]
            totals.append(Ranked(option, total))
        totals.sort(key=lambda r: (-r.total, r.option))
        top = totals[0].total
        margin = top - totals[1].total if len(totals) > 1 else top
        return Decision(tuple(totals), margin)

    # endregion plan_decide
