"""Tests for the flow report."""

from __future__ import annotations

from decimal import Decimal

import pytest

from ai_engineer.leadership.flow import (
    IssueEvent,
    aging,
    cycle_times,
    flow_report,
    littles_check,
    percentile,
    simulate_quarter,
    throughput,
    wip_on,
)


def _issue(name: str, start: int, end: int | None) -> list[IssueEvent]:
    rows = [IssueEvent(name, "todo", 0), IssueEvent(name, "doing", start)]
    if end is not None:
        rows.append(IssueEvent(name, "done", end))
    return rows


EVENTS = (
    _issue("A", 0, 2)
    + _issue("B", 0, 4)
    + _issue("C", 1, 3)
    + _issue("D", 2, 9)
    + _issue("E", 3, None)
)


# region test_flow_basics
def test_flow_basics() -> None:
    assert cycle_times(EVENTS) == {"A": 2, "B": 4, "C": 2, "D": 7}
    assert wip_on(EVENTS, 2) == 3
    assert throughput(EVENTS, 0) == 3
    old = aging(EVENTS, now=8)
    assert [(a.issue_id, a.age) for a in old] == [("D", 6), ("E", 5)]
    assert not any(a.flagged for a in old)
    assert aging(EVENTS, now=14)[0].flagged is True


# endregion test_flow_basics


def test_littles_law_is_exact_for_whole_window() -> None:
    days = 10
    wip = Decimal(sum(wip_on(EVENTS[:-1], d) for d in range(days)))
    cycles = cycle_times(EVENTS)
    check = littles_check(
        wip / days,
        Decimal(len(cycles)) / days,
        Decimal(sum(cycles.values())) / len(cycles),
    )
    assert check.gap == 0
    assert check.agrees


def test_littles_check_flags_disagreement() -> None:
    check = littles_check(Decimal(10), Decimal(1), Decimal(5))
    assert check.predicted_wip == 5
    assert not check.agrees


def test_percentile_nearest_rank() -> None:
    assert percentile([5, 1, 3, 2, 4], 50) == 3
    assert percentile([5, 1, 3, 2, 4], 85) == 5


# region test_flow_refuses_people
@pytest.mark.parametrize("key", ["assignee", "author", "reviewer"])
def test_report_refuses_person_grouping(key: str) -> None:
    with pytest.raises(ValueError, match="never group by"):
        flow_report(EVENTS, now=10, group_by=key)


# endregion test_flow_refuses_people


def test_report_rejects_unknown_grouping_and_empty_data() -> None:
    with pytest.raises(ValueError, match="unknown"):
        flow_report(EVENTS, now=10, group_by="team-color")
    with pytest.raises(ValueError, match="no finished"):
        flow_report(_issue("Z", 0, None), now=3)


def test_report_shape() -> None:
    rep = flow_report(EVENTS, now=10)
    assert rep.p50 == 2
    assert rep.p95 == 7
    assert rep.throughput_by_week == (3, 1)
    assert rep.wip_by_state["doing"] == 1


# region test_wip_limit_cuts_cycle_time
def test_wip_limit_cuts_cycle_time_and_holds_throughput() -> None:
    free = simulate_quarter(wip_limit=6)
    capped = simulate_quarter(wip_limit=2)
    a, b = flow_report(free, now=65), flow_report(capped, now=65)
    assert b.p85 < a.p85
    steady_a = sum(a.throughput_by_week[4:])
    steady_b = sum(b.throughput_by_week[4:])
    assert steady_b >= 0.9 * steady_a


# endregion test_wip_limit_cuts_cycle_time


def test_simulation_is_deterministic() -> None:
    assert simulate_quarter(2) == simulate_quarter(2)
