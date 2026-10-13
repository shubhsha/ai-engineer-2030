"""Flow: make the state of work visible without ranking people.

An issue moves through four states: todo, doing, review and done. An
event records the day it entered a state (working days, counted from
zero). The events carry no person: the data type cannot answer "who is
slow", and ``flow_report`` refuses to group by a person. Cycle time is
the days from the first ``doing`` event to ``done``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from decimal import Decimal

DAYS_PER_WEEK = 5
STATES = ("todo", "doing", "review", "done")
PERSON_KEYS = frozenset(
    {"assignee", "author", "reviewer", "owner", "person", "user", "engineer"}
)
GROUPINGS = frozenset({"state", "week"})
UNITS_PER_DAY = 6  # divisible by 1, 2, 3 and 6 concurrent items


# region flow_events
@dataclass(frozen=True, slots=True)
class IssueEvent:
    issue_id: str
    state: str
    at: int


def _first(events: list[IssueEvent], state: str) -> dict[str, int]:
    found: dict[str, int] = {}
    for e in sorted(events, key=lambda e: e.at):
        if e.state == state and e.issue_id not in found:
            found[e.issue_id] = e.at
    return found


def cycle_times(events: list[IssueEvent]) -> dict[str, int]:
    """Days from first ``doing`` to ``done``, for finished issues."""
    start, end = _first(events, "doing"), _first(events, "done")
    return {i: end[i] - start[i] for i in end if i in start}


def wip_on(events: list[IssueEvent], day: int) -> int:
    """Issues started on or before ``day`` and not yet done."""
    start, end = _first(events, "doing"), _first(events, "done")
    return sum(1 for i, s in start.items() if s <= day < end.get(i, day + 1))


def throughput(events: list[IssueEvent], week: int) -> int:
    """Issues finished in working week ``week`` (zero-based)."""
    lo = week * DAYS_PER_WEEK
    done = _first(events, "done")
    return sum(1 for d in done.values() if lo <= d < lo + DAYS_PER_WEEK)


# endregion flow_events


def percentile(values: list[int], pct: int) -> int:
    """Nearest-rank percentile of a non-empty list."""
    ordered = sorted(values)
    rank = -(-pct * len(ordered) // 100)
    return ordered[max(rank, 1) - 1]


# region flow_aging
@dataclass(frozen=True, slots=True)
class AgingItem:
    issue_id: str
    state: str
    age: int
    flagged: bool


def aging(
    events: list[IssueEvent], now: int, threshold_pct: int = 85
) -> tuple[AgingItem, ...]:
    """Open started issues by age; flag those past the percentile."""
    cycles = list(cycle_times(events).values())
    limit = percentile(cycles, threshold_pct) if cycles else None
    start, end = _first(events, "doing"), _first(events, "done")
    state_now: dict[str, str] = {}
    for e in sorted(events, key=lambda e: e.at):
        if e.at <= now:
            state_now[e.issue_id] = e.state
    items = [
        AgingItem(
            i,
            state_now[i],
            now - s,
            limit is not None and now - s > limit,
        )
        for i, s in start.items()
        if s <= now and (i not in end or end[i] > now)
    ]
    return tuple(sorted(items, key=lambda a: (-a.age, a.issue_id)))


# endregion flow_aging


@dataclass(frozen=True, slots=True)
class LittleCheck:
    predicted_wip: Decimal
    observed_wip: Decimal
    gap: Decimal
    agrees: bool


def littles_check(
    wip: Decimal,
    throughput: Decimal,
    cycle_time: Decimal,
    tolerance: Decimal = Decimal("0.05"),
) -> LittleCheck:
    """Compare observed WIP with throughput times cycle time."""
    predicted = throughput * cycle_time
    gap = abs(predicted - wip) / wip if wip else Decimal(0)
    return LittleCheck(predicted, wip, gap, gap <= tolerance)


# region flow_report
@dataclass(frozen=True, slots=True)
class FlowReport:
    p50: int
    p85: int
    p95: int
    wip_by_state: dict[str, int]
    throughput_by_week: tuple[int, ...]
    flagged: tuple[AgingItem, ...]


def flow_report(
    events: list[IssueEvent], now: int, group_by: str = "state"
) -> FlowReport:
    """Team-level flow numbers. Person-level grouping is refused."""
    if group_by in PERSON_KEYS:
        raise ValueError(f"flow reports never group by {group_by}")
    if group_by not in GROUPINGS:
        raise ValueError(f"unknown grouping: {group_by}")
    cycles = list(cycle_times(events).values())
    if not cycles:
        raise ValueError("no finished issues yet")
    open_items = aging(events, now)
    by_state = dict.fromkeys(STATES[1:3], 0)
    for item in open_items:
        by_state[item.state] = by_state.get(item.state, 0) + 1
    weeks = now // DAYS_PER_WEEK
    return FlowReport(
        percentile(cycles, 50),
        percentile(cycles, 85),
        percentile(cycles, 95),
        by_state,
        tuple(throughput(events, w) for w in range(weeks)),
        tuple(a for a in open_items if a.flagged),
    )


# endregion flow_report


# region flow_simulate
def simulate_quarter(
    wip_limit: int,
    seed: int = 2030,
    people: int = 7,
    days: int = 65,
    arrival_p: float = 0.55,
) -> list[IssueEvent]:
    """A synthetic quarter. Each person splits a day across their items."""
    rng = random.Random(seed)
    events: list[IssueEvent] = []
    backlog: list[tuple[str, int]] = []
    active: list[list[tuple[str, int]]] = [[] for _ in range(people)]
    waiting: list[str] = []
    n = 0
    for day in range(days):
        for _ in range(4):
            if rng.random() < arrival_p:
                n += 1
                size = rng.choice((1, 2, 2, 3, 3, 4, 5)) * UNITS_PER_DAY
                events.append(IssueEvent(f"I{n:03d}", "todo", day))
                backlog.append((f"I{n:03d}", size))
        for _id in waiting:
            events.append(IssueEvent(_id, "done", day))
        waiting = []
        for mine in active:
            while backlog and len(mine) < wip_limit:
                issue_id, size = backlog.pop(0)
                events.append(IssueEvent(issue_id, "doing", day))
                mine.append((issue_id, size))
            budget = UNITS_PER_DAY
            for k in range(len(mine)):
                share = budget // (len(mine) - k)
                budget -= share
                mine[k] = (mine[k][0], mine[k][1] - share)
            for issue_id, _left in [m for m in mine if m[1] <= 0]:
                events.append(IssueEvent(issue_id, "review", day + 1))
                waiting.append(issue_id)
            mine[:] = [m for m in mine if m[1] > 0]
    return events


# endregion flow_simulate
