from __future__ import annotations

from datetime import date

import pytest

from ai_engineer.career.manual import Entry, Manual, Verdict, weigh


def _manual() -> Manual:
    m = Manual()
    m = m.add(Entry(date(2026, 1, 5), "Mornings", "Hard work first."))
    m = m.add(Entry(date(2026, 2, 2), "Reviews", "Read every diff."))
    return m


# region test_current_hides_superseded
def test_current_hides_superseded() -> None:
    m = _manual().add(
        Entry(
            date(2026, 3, 1),
            "Mornings v2",
            "Hard work after a walk.",
            status="accepted",
            supersedes="Mornings",
        )
    )
    titles = [e.title for e in m.current()]
    assert titles == ["Reviews", "Mornings v2"]
    assert len(m.entries) == 3


# endregion


def test_status_superseded_is_hidden() -> None:
    m = _manual().add(Entry(date(2026, 3, 1), "Old", "x", status="superseded"))
    assert "Old" not in [e.title for e in m.current()]


def test_add_rejects_bad_entries() -> None:
    m = _manual()
    with pytest.raises(ValueError, match="duplicate"):
        m.add(Entry(date(2026, 4, 1), "Mornings", "again"))
    with pytest.raises(ValueError, match="date order"):
        m.add(Entry(date(2025, 1, 1), "Early", "x"))
    with pytest.raises(ValueError, match="unknown"):
        m.add(Entry(date(2026, 4, 1), "New", "x", supersedes="Nope"))


def test_add_does_not_mutate() -> None:
    m = _manual()
    m.add(Entry(date(2026, 4, 1), "New", "x"))
    assert len(m.entries) == 2


def test_stale_lists_entries_not_revised() -> None:
    m = _manual()
    assert [e.title for e in m.stale(date(2026, 4, 1), days=60)] == [
        "Mornings"
    ]
    assert m.stale(date(2026, 1, 20)) == ()


# region test_weigh
def test_weigh_votes_for_the_heavier_side() -> None:
    assert weigh([3, 2, 1], [2, 1]) == Verdict("for", 3)
    assert weigh([1], [4, 2]) == Verdict("against", 5)
    assert weigh([2, 2], [4]) == Verdict("tie", 0)


# endregion


def test_weigh_rejects_negative_points() -> None:
    with pytest.raises(ValueError, match="negative"):
        weigh([-1], [1])
