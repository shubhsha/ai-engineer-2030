from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from ai_engineer.career.transition import (
    Criterion,
    Experiment,
    OrderError,
    Scorecard,
    overdue,
    runway_months,
    total_cost,
)

D = Decimal


def test_runway_with_a_pay_cut() -> None:
    assert runway_months(D(24000), D(3000), D(0)) == D("8.0")
    assert runway_months(D(24000), D(3000), D(1800)) == D("20.0")


def test_runway_is_none_when_income_covers_costs() -> None:
    assert runway_months(D(5000), D(3000), D(3000)) is None


def test_runway_rejects_negative_inputs() -> None:
    with pytest.raises(ValueError, match="negative"):
        runway_months(D(-1), D(3000), D(0))


def test_experiments_cost_and_deadlines() -> None:
    a = Experiment("Do I like on-call?", D(0), date(2026, 11, 1))
    b = Experiment("Can I pass the exam?", D(300), date(2026, 12, 1))
    assert total_cost([a, b]) == D(300)
    assert overdue([a, b], date(2026, 11, 2)) == (a,)
    assert overdue([a, b], date(2026, 11, 1)) == ()


def _fixed() -> Scorecard:
    card = Scorecard()
    card.fix_criteria(
        [
            Criterion("learning", D(3)),
            Criterion("pay", D(2)),
            Criterion("reversibility", D(3)),
        ]
    )
    return card


def test_scores_before_criteria_are_refused() -> None:
    with pytest.raises(OrderError):
        Scorecard().score("stay", "pay", 3)


def test_criteria_cannot_change_after_scoring() -> None:
    card = _fixed()
    card.score("stay", "pay", 3)
    with pytest.raises(OrderError):
        card.fix_criteria([Criterion("reversibility", D(1))])


def test_reversibility_criterion_is_required() -> None:
    with pytest.raises(ValueError, match="required"):
        Scorecard().fix_criteria([Criterion("pay", D(1))])


def test_decide_ranks_and_reports_margin() -> None:
    card = _fixed()
    for name, vals in {"stay": (2, 4, 5), "move": (5, 3, 2)}.items():
        for c, v in zip(
            ("learning", "pay", "reversibility"), vals, strict=True
        ):
            card.score(name, c, v)
    result = card.decide()
    assert [r.option for r in result.ranked] == ["stay", "move"]
    assert result.ranked[0].total == D(29)
    assert result.ranked[1].total == D(27)
    assert result.winner == "stay"
    assert result.margin == D(2)


def test_missing_score_and_bad_value() -> None:
    card = _fixed()
    card.score("stay", "pay", 3)
    with pytest.raises(ValueError, match="lacks"):
        card.decide()
    with pytest.raises(ValueError, match="0 to 5"):
        card.score("stay", "pay", 9)
