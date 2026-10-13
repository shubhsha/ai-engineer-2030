"""Leverage: where an engineer's hours produce the most.

An activity is a block of hours with a reach (people affected) and a
life (weeks the output stays useful). Its impact is reach times life,
in person-weeks, and its leverage is impact per hour. The numbers are
crude on purpose: they make the trade-offs of a week visible so that
a person can argue with them.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal

GLUE_KINDS = frozenset(
    {"one-to-ones", "runbooks", "coordination", "incident help"}
)


# region leverage_activity
@dataclass(frozen=True, slots=True)
class Activity:
    kind: str
    hours: Decimal
    people_affected: int
    durability_weeks: int
    visible: bool

    @property
    def impact(self) -> int:
        """Person-weeks touched: people affected times weeks it lasts."""
        return self.people_affected * self.durability_weeks


@dataclass(frozen=True, slots=True)
class KindRow:
    kind: str
    hours: Decimal
    impact: int
    leverage: Decimal  # person-weeks per hour


# endregion


# region week_ledger
def week_ledger(activities: list[Activity]) -> tuple[KindRow, ...]:
    """Hours, impact and leverage per kind, highest leverage first."""
    hours: dict[str, Decimal] = defaultdict(Decimal)
    impact: dict[str, int] = defaultdict(int)
    for a in activities:
        if a.hours <= 0:
            raise ValueError(f"{a.kind}: hours must be positive")
        hours[a.kind] += a.hours
        impact[a.kind] += a.impact
    rows = [
        KindRow(k, hours[k], impact[k], Decimal(impact[k]) / hours[k])
        for k in hours
    ]
    return tuple(sorted(rows, key=lambda r: (-r.leverage, r.kind)))


def _share(hours: Decimal, total: Decimal) -> Decimal:
    return hours / total if total else Decimal(0)


def glue_share(
    activities: list[Activity], glue_kinds: frozenset[str] = GLUE_KINDS
) -> Decimal:
    """Share of the hours spent on glue kinds."""
    total = sum((a.hours for a in activities), Decimal(0))
    glue = sum(
        (a.hours for a in activities if a.kind in glue_kinds), Decimal(0)
    )
    return _share(glue, total)


def invisible_share(activities: list[Activity]) -> Decimal:
    """Share of the hours that nobody outside the work can see."""
    total = sum((a.hours for a in activities), Decimal(0))
    hidden = sum((a.hours for a in activities if not a.visible), Decimal(0))
    return _share(hidden, total)


# endregion


# region recommend_move
@dataclass(frozen=True, slots=True)
class Move:
    from_kind: str
    to_kind: str
    hours: Decimal
    gain: Decimal  # person-weeks per week
    reason: str


def recommend_move(
    activities: list[Activity],
    hours: Decimal = Decimal(4),
    discount: Decimal = Decimal("0.5"),
) -> Move:
    """Move hours from the lowest-leverage kind to the highest.

    The freed hours cost the source kind's average leverage. The new
    hours earn only `discount` of the target's average, because the
    next hour of anything is worth less than the average one. At most
    half of the source kind's hours move, and no more than the target
    already has: you cannot more than double a kind in a week.
    """
    rows = week_ledger(activities)
    if len(rows) < 2:
        raise ValueError("need at least two kinds of activity")
    top, low = rows[0], rows[-1]
    moved = min(hours, low.hours / 2, top.hours)
    gain = moved * (top.leverage * discount - low.leverage)
    reason = (
        f"{low.kind} returns {low.leverage:.1f} person-weeks per hour; "
        f"{top.kind} returns {top.leverage:.1f}"
    )
    return Move(low.kind, top.kind, moved, gain, reason)


def weekly_impact(activities: list[Activity]) -> int:
    """Total person-weeks touched by the week; after a move, add its gain."""
    return sum(a.impact for a in activities)


# endregion


# region queue_and_meetings
def channels(n: int) -> int:
    """Pairs of people on a team of n: n(n-1)/2, Brooks's arithmetic."""
    return n * (n - 1) // 2


def mean_in_queue(arrivals: Decimal, capacity: Decimal) -> Decimal:
    """Mean number of items waiting or in service, M/M/1: rho/(1-rho)."""
    rho = arrivals / capacity
    if rho >= 1:
        raise ValueError("arrivals meet or exceed capacity: queue grows")
    return rho / (1 - rho)


def meeting_cost(people: int, hours: Decimal, hourly_rate: Decimal) -> Decimal:
    """People times hours times a loaded hourly rate."""
    return people * hours * hourly_rate


# endregion
