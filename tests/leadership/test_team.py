"""Tests for the survey aggregator, the bootstrap and delegation."""

from __future__ import annotations

import random

import pytest

from ai_engineer.leadership.team import (
    Delegation,
    DelegationLevel,
    SurveyResponse,
    aggregate,
    bootstrap_interval,
    change_is_clear,
)


def _rows(group: str, n: int, score: int) -> list[SurveyResponse]:
    return [SurveyResponse("safe", score, "Q1", group) for _ in range(n)]


def test_small_group_is_suppressed() -> None:
    rows = _rows("a", 4, 5) + _rows("b", 5, 3)
    out = aggregate(rows)
    assert [r.group for r in out] == ["b"]
    assert out[0].n == 5
    assert out[0].mean == 3.0


def test_floor_is_configurable() -> None:
    assert len(aggregate(_rows("a", 3, 5), floor=3)) == 1


def test_score_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match="1 to 7"):
        SurveyResponse("safe", 8, "Q1", "a")


def test_interval_is_seeded_and_contains_mean(rng: random.Random) -> None:
    scores = [4.0, 5.0, 6.0, 5.0, 4.0, 6.0, 5.0, 5.0]
    lo, hi = bootstrap_interval(scores, rng)
    assert lo <= 5.0 <= hi
    again = bootstrap_interval(scores, random.Random(2030))
    assert again == (lo, hi)


def test_interval_of_constant_scores_is_a_point(rng: random.Random) -> None:
    assert bootstrap_interval([3.0] * 6, rng) == (3.0, 3.0)


def test_empty_scores_rejected(rng: random.Random) -> None:
    with pytest.raises(ValueError, match="no scores"):
        bootstrap_interval([], rng)


def test_change_is_clear_needs_disjoint_intervals() -> None:
    assert change_is_clear((3.0, 4.0), (4.5, 5.5))
    assert change_is_clear((4.5, 5.5), (3.0, 4.0))
    assert not change_is_clear((3.0, 4.6), (4.5, 5.5))


def test_delegation_level_decides() -> None:
    d = Delegation("pick a queue", "ana", DelegationLevel.CONSULT, "Fri")
    assert not d.decides("ana")
    up = Delegation("pick a queue", "ana", DelegationLevel.ADVISE, "Fri")
    assert up.decides("ana")
    assert not up.decides("lead")
    assert DelegationLevel.INFORM > DelegationLevel.TELL
