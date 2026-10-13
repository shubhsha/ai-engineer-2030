from __future__ import annotations

import random
from datetime import date

import pytest

from ai_engineer.career.learning import (
    Card,
    due_today,
    plan_year,
    retrievability,
    review,
    simulate,
)

D0 = date(2030, 1, 1)


def test_sm2_intervals_for_passing_answers() -> None:
    card = Card("q", "a", D0)
    days = []
    for _ in range(4):
        card = review(card, 4, D0)
        days.append(card.interval)
    assert days == [1, 6, 15, 38]


def test_perfect_answer_raises_ease_and_failure_resets() -> None:
    card = review(Card("q", "a", D0), 5, D0)
    assert card.ease == pytest.approx(2.6)
    failed = review(card, 1, D0)
    assert (failed.reps, failed.interval) == (0, 1)
    assert failed.due == date(2030, 1, 2)


def test_ease_never_falls_below_floor() -> None:
    card = Card("q", "a", D0, ease=1.3)
    assert review(card, 3, D0).ease == 1.3


def test_bad_quality_is_rejected() -> None:
    with pytest.raises(ValueError, match="quality"):
        review(Card("q", "a", D0), 6, D0)


def test_due_today_orders_overdue_first() -> None:
    cards = [
        Card("b", "", date(2030, 1, 3)),
        Card("a", "", date(2030, 1, 5)),
        Card("c", "", date(2030, 1, 1)),
        Card("z", "", date(2030, 2, 1)),
    ]
    got = due_today(cards, date(2030, 1, 5))
    assert [c.front for c in got] == ["c", "b", "a"]


def test_plan_takes_four_largest_gaps_with_practice() -> None:
    gaps = {
        "writing": 3,
        "speaking": 2,
        "sql": 2,
        "domain": 1,
        "go": 0,
        "ops": 1,
    }
    plan = plan_year(gaps, D0)
    assert [q.skill for q in plan] == ["writing", "speaking", "sql", "domain"]
    assert all(q.project and q.practice for q in plan)
    assert plan[1].start == date(2030, 4, 2)


def test_plan_rejects_empty_gaps() -> None:
    with pytest.raises(ValueError, match="no gaps"):
        plan_year({}, D0)


def test_retrievability_decays() -> None:
    assert retrievability(0, 3) == 1.0
    assert retrievability(3, 3) == pytest.approx(0.3679, abs=1e-4)


def test_spaced_beats_massed_on_a_seeded_run(rng: random.Random) -> None:
    spaced, reviews = simulate(rng, True, n_cards=60)
    massed, _ = simulate(
        random.Random(2030), False, 60, reviews=round(reviews)
    )
    assert spaced > massed
