"""Tests for incident leadership."""

from __future__ import annotations

import pytest

from ai_engineer.leadership.incident import (
    ActionItem,
    Event,
    IllegalTransition,
    Incident,
    Role,
    Severity,
    State,
    assign_roles,
    missed_updates,
    open_items,
    overdue,
    simulate,
    timeline,
)


def test_missed_updates_counts_silent_windows() -> None:
    # SEV1 every 15 minutes, open 0..50: updates at 10 and 40.
    # Deadlines 15 (met by 10), 25 (missed), 40 (met), 55 (after close).
    assert missed_updates([10, 40], Severity.SEV1, 0, 50) == 1
    assert missed_updates([], Severity.SEV1, 0, 50) == 3
    assert missed_updates([], Severity.SEV1, 0, 15) == 0


def test_cadence_depends_on_severity() -> None:
    assert missed_updates([], Severity.SEV3, 0, 60) == 0
    assert missed_updates([], Severity.SEV2, 0, 61) == 2


def test_walk_through_all_states() -> None:
    inc = Incident("db", Severity.SEV1, declared_at=0)
    inc = inc.transition(State.MITIGATED, 12)
    inc = inc.transition(State.RESOLVED, 40)
    inc = inc.transition(State.REVIEWED, 2000)
    assert inc.minutes_to(State.MITIGATED) == 12
    assert inc.minutes_to(State.RESOLVED) == 40


def test_illegal_moves_are_refused() -> None:
    inc = Incident("db", Severity.SEV2, declared_at=10)
    with pytest.raises(IllegalTransition):
        inc.transition(State.RESOLVED, 20)
    with pytest.raises(IllegalTransition):
        inc.transition(State.MITIGATED, 5)
    mitigated = inc.transition(State.MITIGATED, 20)
    with pytest.raises(IllegalTransition):
        mitigated.transition(State.DECLARED, 30)


def test_review_blocked_by_open_items() -> None:
    item = ActionItem("add alert", "ana", due=100)
    inc = Incident("db", Severity.SEV3, 0, items=(item,))
    inc = inc.transition(State.MITIGATED, 5).transition(State.RESOLVED, 9)
    with pytest.raises(IllegalTransition):
        inc.transition(State.REVIEWED, 50)
    done = ActionItem("add alert", "ana", due=100, done=True)
    ok = Incident("db", Severity.SEV3, 0, State.RESOLVED, items=(done,))
    assert ok.transition(State.REVIEWED, 50).state is State.REVIEWED


def test_commander_cannot_hold_operations() -> None:
    with pytest.raises(ValueError, match="does not hold"):
        assign_roles({Role.COMMANDER: "ana", Role.OPERATIONS: "ana"})
    with pytest.raises(ValueError, match="needs a commander"):
        assign_roles({Role.OPERATIONS: "ben"})
    ok = assign_roles({Role.COMMANDER: "ana", Role.OPERATIONS: "ben"})
    assert ok[Role.OPERATIONS] == "ben"


def test_open_and_overdue_items() -> None:
    items = (
        ActionItem("b", "x", due=20),
        ActionItem("a", "y", due=10),
        ActionItem("c", "z", due=5, done=True),
    )
    assert [i.title for i in open_items(items)] == ["a", "b"]
    assert [i.title for i in overdue(items, now=15)] == ["a"]


def test_timeline_is_sorted_with_offsets() -> None:
    rows = timeline(
        [Event(12, "mitigate", "rolled back"), Event(3, "declare", "SEV1")],
        declared_at=3,
    )
    assert rows[0].startswith("+0 declare")
    assert rows[1].startswith("+9 mitigate")


def test_simulation_is_seeded_and_favors_a_commander() -> None:
    a, b = simulate(2030), simulate(2030)
    assert a == b
    assert a.with_ic < a.without_ic
    assert a.missed_with < a.missed_without
    assert set(a.by_severity) == {"SEV1", "SEV2", "SEV3"}
