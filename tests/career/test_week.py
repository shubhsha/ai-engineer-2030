from __future__ import annotations

import random

from ai_engineer.career.week import (
    Commitment,
    Week,
    fragmentation,
    plan,
    review,
    simulate,
)

WEEK = Week(capacity=40, meetings=14, buffer=6)
ITEMS = [
    Commitment("design review", 8, 6),
    Commitment("migration doc", 5, 10),
    Commitment("eval set", 9, 8),
    Commitment("hiring loop", 4, 4),
]


def test_available_hours_are_what_remains_after_meetings_and_buffer() -> None:
    assert WEEK.available == 20


def test_plan_takes_highest_impact_per_hour_and_drops_what_does_not_fit() -> (
    None
):
    got = plan(WEEK, ITEMS)
    assert [c.title for c in got.chosen] == [
        "design review",
        "eval set",
        "hiring loop",
    ]
    assert [c.title for c in got.dropped] == ["migration doc"]
    assert got.planned_hours == 18
    assert got.free == 2


def test_a_plan_never_exceeds_available_hours() -> None:
    got = plan(Week(40, 30, 6), ITEMS)
    assert got.planned_hours <= 4


def test_fragmentation_counts_only_blocks_of_ninety_minutes_or_more() -> None:
    assert fragmentation([30, 90, 45, 120, 89]) == 2
    assert fragmentation([30, 60], min_minutes=60) == 1


def test_review_reports_the_estimate_ratio_and_moves_the_buffer() -> None:
    got = review([4, 6], [6, 9], interruptions=8, buffer=4)
    assert got.ratio == 1.5
    assert got.next_buffer == 6


def test_review_of_an_empty_week_is_neutral() -> None:
    assert review([], [], 0, 4).ratio == 1.0


def test_the_correction_shrinks_overcommitment() -> None:
    raw = simulate(random.Random(2030), correct=False)
    fixed = simulate(random.Random(2030), correct=True)
    assert len(raw) == len(fixed) == 12
    assert sum(fixed) < sum(raw)
