from __future__ import annotations

from decimal import Decimal

import pytest

from ai_engineer.leadership.leverage import (
    Activity,
    channels,
    glue_share,
    invisible_share,
    mean_in_queue,
    meeting_cost,
    recommend_move,
    week_ledger,
    weekly_impact,
)

D = Decimal


# region composite_week
def composite_week() -> list[Activity]:
    """A 45-hour week of a staff engineer (a composite)."""
    return [
        Activity("coding", D(17), 4, 12, True),
        Activity("reviews", D(8), 6, 8, True),
        Activity("design docs", D(4), 40, 26, True),
        Activity("one-to-ones", D(3), 5, 12, False),
        Activity("meetings", D(6), 8, 1, True),
        Activity("incident help", D(4), 30, 2, True),
        Activity("runbooks", D(1), 25, 52, False),
        Activity("coordination", D(2), 6, 2, False),
    ]


# endregion


def test_hours_add_to_45() -> None:
    rows = week_ledger(composite_week())
    assert sum(r.hours for r in rows) == 45


def test_ledger_sorted_by_leverage() -> None:
    rows = week_ledger(composite_week())
    assert [r.kind for r in rows][:2] == ["runbooks", "design docs"]
    assert rows[0].leverage == 1300
    assert rows[1].leverage == 260
    assert rows[-1].kind == "meetings"
    assert rows[-1].leverage == D(8) / D(6)


def test_shares() -> None:
    week = composite_week()
    assert glue_share(week) == D(10) / D(45)
    assert invisible_share(week) == D(6) / D(45)
    assert glue_share([]) == 0


def test_recommend_move_gains() -> None:
    week = composite_week()
    move = recommend_move(week)
    assert (move.from_kind, move.to_kind) == ("meetings", "runbooks")
    assert move.hours == 1  # the target has one hour: it can only double
    assert move.gain == 1 * (D(1300) / 2 - D(8) / 6)
    assert move.gain > 0
    assert "meetings" in move.reason
    assert weekly_impact(week) == 48 + 48 + 1040 + 60 + 8 + 60 + 1300 + 12


def test_recommend_needs_two_kinds() -> None:
    with pytest.raises(ValueError, match=r"\w"):
        recommend_move([Activity("coding", D(1), 1, 1, True)])


def test_bad_hours() -> None:
    with pytest.raises(ValueError, match=r"\w"):
        week_ledger([Activity("x", D(0), 1, 1, True)])


def test_brooks_channels() -> None:
    assert [channels(n) for n in (3, 5, 10, 11)] == [3, 10, 45, 55]


def test_queue_blows_up_near_capacity() -> None:
    assert mean_in_queue(D(8), D(10)) == 4
    assert mean_in_queue(D("9.5"), D(10)) == 19
    with pytest.raises(ValueError, match=r"\w"):
        mean_in_queue(D(10), D(10))


def test_meeting_cost() -> None:
    assert meeting_cost(8, D(1), D(100)) == 800
