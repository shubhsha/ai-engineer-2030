"""Tests for the skills matrix and the 90-day plan."""

from __future__ import annotations

import random
from datetime import date

from ai_engineer.leadership.growth import (
    Level,
    SkillRow,
    follow_plan,
    gaps,
    plan_90_days,
    team_gaps,
)

N, B, C, P, E = (
    Level.NOVICE,
    Level.ADVANCED_BEGINNER,
    Level.COMPETENT,
    Level.PROFICIENT,
    Level.EXPERT,
)

MATRIX = [
    SkillRow("design", B, P),
    SkillRow("review", C, P),
    SkillRow("testing", C, C),
    SkillRow("writing", N, B),
    SkillRow("operations", B, C),
]


def test_gap_is_never_negative() -> None:
    assert SkillRow("x", E, C).gap == 0
    assert SkillRow("x", B, P).gap == 2


def test_gaps_rank_by_size_times_need() -> None:
    ranked = gaps(MATRIX, {"review": 5})
    assert [g.skill for g in ranked] == [
        "review",
        "design",
        "operations",
        "writing",
    ]
    assert all(g.size > 0 for g in ranked)


def test_team_gaps_sum_over_people() -> None:
    team = {"a": MATRIX, "b": [SkillRow("design", N, C)]}
    top = team_gaps(team, {})[0]
    assert (top.skill, top.size) == ("design", 4)


def test_plan_has_at_most_two_items_with_dates() -> None:
    plan = plan_90_days("Ana", MATRIX, {}, date(2030, 1, 7))
    assert len(plan) == 2
    assert [i.check_in for i in plan] == [
        date(2030, 2, 21),
        date(2030, 4, 7),
    ]
    assert "design" in plan[0].stretch


def test_plan_is_empty_when_nothing_is_open() -> None:
    done = [SkillRow("x", E, E)]
    assert plan_90_days("Ana", done, {}, date(2030, 1, 7)) == ()


def test_follow_plan_only_moves_planned_skills(rng: random.Random) -> None:
    plan = plan_90_days("Ana", MATRIX, {}, date(2030, 1, 7))
    after = follow_plan(MATRIX, plan, rng, p_climb=1.0)
    moved = {a.skill for a, b in zip(after, MATRIX, strict=True) if a != b}
    assert moved == {i.gap for i in plan}
    assert all(a.current <= a.target for a in after)
