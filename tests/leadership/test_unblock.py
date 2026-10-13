"""Tests for the blocker log."""

from __future__ import annotations

import random

import pytest

from ai_engineer.leadership.unblock import (
    Blocker,
    aging,
    littles_check,
    percentile,
    simulate_quarter,
    time_to_unblock,
    wait_graph,
    wait_share,
)

LOG = [
    Blocker(0, "ana", "lee", 4),
    Blocker(2, "bo", "lee", 12),
    Blocker(3, "cy", "dee", 5),
    Blocker(8, "ana", "lee"),
]


def test_cannot_clear_before_raised() -> None:
    with pytest.raises(ValueError, match="before it is raised"):
        Blocker(5, "a", "b", 4)


def test_percentile_is_nearest_rank() -> None:
    assert percentile([1, 2, 3, 4], 50) == 2
    assert percentile([1, 2, 3, 4], 90) == 4


def test_time_to_unblock_ignores_open() -> None:
    out = time_to_unblock(LOG)
    assert out["p50"] == 4
    assert out["mean"] == pytest.approx((4 + 10 + 2) / 3)


def test_aging_lists_open_oldest_first() -> None:
    rows = aging(LOG, now=10)
    assert [(b.waiting, age) for b, age in rows] == [("ana", 2)]


def test_wait_graph_ranks_the_most_waited_on() -> None:
    ranked = wait_graph(LOG, now=10)
    assert ranked[0] == ("lee", 14.0)
    assert wait_share(LOG, 10, "lee") == pytest.approx(14 / 16)
    assert wait_share(LOG, 10, "nobody") == 0.0


def test_littles_law_holds_on_a_seeded_quarter(rng: random.Random) -> None:
    log = simulate_quarter(rng, ["a", "b", "c"], [3, 1, 1], 0.2, 5.0)
    check = littles_check(log, 0, 520)
    assert check.predicted_open == pytest.approx(check.observed_open)
    assert 0.1 < check.arrival_rate < 0.3


def test_littles_check_needs_arrivals() -> None:
    with pytest.raises(ValueError, match="arrivals"):
        littles_check([], 0, 10)
