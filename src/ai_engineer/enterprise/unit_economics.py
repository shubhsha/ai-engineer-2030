"""Chapter 51: what one successful task of an AI feature costs and earns.

A feature is a list of per-attempt costs, a success rate and a price per
successful task. Every other number in the chapter (margin, break-even,
the tornado chart) is arithmetic on those three things. Money is
``Decimal`` throughout; rates are ``Decimal`` too, so nothing rounds
until a person reads it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from decimal import Decimal

from ai_engineer.claude.catalog import ModelInfo
from ai_engineer.claude.cost import call_cost
from ai_engineer.enterprise.evidence import Evidence
from ai_engineer.llm.model import Usage

ZERO = Decimal(0)
ONE = Decimal(1)
KINDS = ("tokens", "review", "other")


# region cost_model
@dataclass(frozen=True, slots=True)
class CostComponent:
    """One line of the cost stack, in dollars per attempt.

    ``kind`` groups lines for sensitivity: ``tokens`` scales with prompt
    and reply size, ``review`` with the share of work a person checks.
    """

    name: str
    per_attempt: Decimal
    kind: str = "other"


@dataclass(frozen=True, slots=True)
class UnitModel:
    """A feature's unit: costs per attempt, success rate, price per win."""

    components: tuple[CostComponent, ...]
    success_rate: Decimal
    price: Decimal

    def __post_init__(self) -> None:
        if not ZERO < self.success_rate <= ONE:
            raise ValueError("success_rate must be in (0, 1]")


def token_component(
    model: ModelInfo, calls: tuple[Usage, ...], *, batch: bool = False
) -> CostComponent:
    """The token line for one attempt, priced by ``call_cost``."""
    total = sum((call_cost(model, u, batch=batch) for u in calls), ZERO)
    return CostComponent("tokens", total, "tokens")


def review_component(
    share: Decimal, minutes: Decimal, hourly_rate: Decimal
) -> CostComponent:
    """Human review: the share of attempts a person reads, times their time."""
    return CostComponent(
        "review", share * minutes / Decimal(60) * hourly_rate, "review"
    )


def cost_per_attempt(model: UnitModel) -> Decimal:
    return sum((c.per_attempt for c in model.components), ZERO)


def cost_per_success(model: UnitModel) -> Decimal:
    """Every attempt is paid for; only the successes are sold."""
    return cost_per_attempt(model) / model.success_rate


def cost_stack(model: UnitModel) -> dict[str, Decimal]:
    """Each line, restated per successful task. The lines sum to the total."""
    return {
        c.name: c.per_attempt / model.success_rate for c in model.components
    }


# endregion


# region margin
@dataclass(frozen=True, slots=True)
class Margin:
    contribution: Decimal  # dollars left from one successful task
    ratio: Decimal  # contribution as a share of price


def margin(model: UnitModel) -> Margin:
    contribution = model.price - cost_per_success(model)
    ratio = contribution / model.price if model.price else ZERO
    return Margin(contribution, ratio)


def break_even_price(model: UnitModel) -> Decimal:
    """The price at which one successful task earns exactly what it costs."""
    return cost_per_success(model)


def break_even_volume(model: UnitModel, fixed_per_month: Decimal) -> int:
    """Successful tasks per month needed to cover the fixed costs."""
    contribution = margin(model).contribution
    if contribution <= ZERO:
        raise ValueError("no volume breaks even at a non-positive margin")
    return math.ceil(fixed_per_month / contribution)


def crossover_volume(
    api_per_task: Decimal, hosted_fixed: Decimal, hosted_per_task: Decimal
) -> int | None:
    """Tasks per month above which self-hosting is the cheaper way."""
    saving = api_per_task - hosted_per_task
    if saving <= ZERO:
        return None
    return math.ceil(hosted_fixed / saving)


# endregion


# region sensitivity
@dataclass(frozen=True, slots=True)
class TornadoRow:
    """Margin per successful task when one input moves down and up."""

    name: str
    low: Decimal
    high: Decimal

    @property
    def swing(self) -> Decimal:
        return abs(self.high - self.low)


def _scaled(model: UnitModel, name: str, factor: Decimal) -> UnitModel:
    if name == "success_rate":
        rate = min(ONE, model.success_rate * factor)
        return replace(model, success_rate=rate)
    if name == "price":
        return replace(model, price=model.price * factor)
    if name not in KINDS:
        raise KeyError(name)
    lines = tuple(
        replace(c, per_attempt=c.per_attempt * factor) if c.kind == name else c
        for c in model.components
    )
    return replace(model, components=lines)


def sensitivity(
    model: UnitModel, deltas: Mapping[str, Decimal]
) -> tuple[TornadoRow, ...]:
    """Move each input by ``-d`` and ``+d`` (relative), one at a time.

    Rows come back widest swing first, which is the order a tornado
    chart draws them in.
    """
    rows = []
    for name, d in deltas.items():
        down = margin(_scaled(model, name, ONE - d)).contribution
        up = margin(_scaled(model, name, ONE + d)).contribution
        rows.append(TornadoRow(name, down, up))
    return tuple(sorted(rows, key=lambda r: r.swing, reverse=True))


# endregion


# region budget
def budget_evidence(
    feature: str,
    model: UnitModel,
    budget: Decimal,
    *,
    run_id: str,
    recorded_at: float,
) -> Evidence:
    """One evidence record: is the cost per success within its budget?"""
    cost = cost_per_success(model)
    return Evidence(
        test_id=f"unit-cost:{feature}",
        run_id=run_id,
        recorded_at=recorded_at,
        scope=feature,
        expected=f"cost per success <= {budget:.4f}",
        observed=f"{cost:.4f}",
        verdict="pass" if cost <= budget else "fail",
        location="ai_engineer/enterprise/unit_economics.py",
    )


# endregion
