"""Chapter 67: the money arithmetic behind a staff engineer's case.

Net present value and payback, build versus buy as total cost of
ownership, cost of delay ordered by CD3, and technical debt as interest.
All money is ``Decimal``; every function is pure.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from ai_engineer.claude.catalog import ModelInfo
from ai_engineer.claude.cost import call_cost
from ai_engineer.llm.model import Usage

CENT = Decimal("0.01")


# region npv
def monthly_rate(annual: Decimal) -> Decimal:
    """The monthly rate that compounds to ``annual`` over 12 months."""
    return (1 + annual) ** (Decimal(1) / 12) - 1


def npv(rate: Decimal, flows: list[Decimal]) -> Decimal:
    """Net present value. ``flows[0]`` happens now; ``flows[t]`` at the
    end of period ``t``; ``rate`` is the rate per period."""
    total = Decimal(0)
    for t, flow in enumerate(flows):
        total += flow / (1 + rate) ** t
    return total.quantize(CENT)


def payback_month(flows: list[Decimal]) -> int | None:
    """First period where the running total reaches zero, else None."""
    running = Decimal(0)
    for t, flow in enumerate(flows):
        running += flow
        if t > 0 and running >= 0:
            return t
    return None


# endregion npv


# region tco
@dataclass(frozen=True, slots=True)
class Tco:
    """What one option costs: once, every year, and on the way out."""

    name: str
    upfront: Decimal
    yearly: Decimal
    exit_cost: Decimal = Decimal(0)

    def total(self, years: int) -> Decimal:
        return self.upfront + self.yearly * years + self.exit_cost


@dataclass(frozen=True, slots=True)
class Verdict:
    build: Decimal
    buy: Decimal
    cheaper: str
    gap: Decimal


def build_vs_buy(build: Tco, buy: Tco, years: int) -> Verdict:
    a, b = build.total(years), buy.total(years)
    cheaper = build.name if a < b else buy.name
    return Verdict(a, b, cheaper, abs(a - b))


def model_spend_per_year(
    model: ModelInfo, usage: Usage, calls_per_year: int
) -> Decimal:
    """Yearly token spend, using the book's call_cost."""
    return call_cost(model, usage) * calls_per_year


# endregion tco


# region delay
@dataclass(frozen=True, slots=True)
class Job:
    name: str
    cost_of_delay_per_week: Decimal
    duration_weeks: Decimal

    @property
    def cd3(self) -> Decimal:
        return self.cost_of_delay_per_week / self.duration_weeks


def order_fifo(jobs: list[Job]) -> list[Job]:
    return list(jobs)


def order_sjf(jobs: list[Job]) -> list[Job]:
    return sorted(jobs, key=lambda j: j.duration_weeks)


def order_cd3(jobs: list[Job]) -> list[Job]:
    return sorted(jobs, key=lambda j: j.cd3, reverse=True)


def total_delay_cost(order: list[Job]) -> Decimal:
    """Each job pays its weekly cost until it finishes; one team, one
    job at a time."""
    clock = Decimal(0)
    total = Decimal(0)
    for job in order:
        clock += job.duration_weeks
        total += job.cost_of_delay_per_week * clock
    return total


# endregion delay


# region debt
def debt_interest(
    changes: int, extra_hours: Decimal, paid_down_at: int | None = None
) -> Decimal:
    """Hours lost to a shortcut: ``extra_hours`` on every change made
    before it was paid down (``paid_down_at`` counts changes)."""
    paying = changes if paid_down_at is None else min(changes, paid_down_at)
    return extra_hours * paying


def breakeven_changes(principal_hours: Decimal, extra_hours: Decimal) -> int:
    """Changes after which paying the debt down beats living with it."""
    whole = int(principal_hours // extra_hours)
    return whole if whole * extra_hours == principal_hours else whole + 1


# endregion debt
