from decimal import Decimal as D

import pytest

from ai_engineer.claude.catalog import load_catalog
from ai_engineer.enterprise.unit_economics import (
    CostComponent,
    UnitModel,
    break_even_price,
    break_even_volume,
    budget_evidence,
    cost_per_attempt,
    cost_per_success,
    cost_stack,
    crossover_volume,
    margin,
    review_component,
    sensitivity,
    token_component,
)
from ai_engineer.llm.model import Usage


def feature(rate="0.8", price="1.50"):
    return UnitModel(
        (
            CostComponent("tokens", D("0.03"), "tokens"),
            review_component(D("0.2"), D(3), D(60)),
        ),
        D(rate),
        D(price),
    )


def test_the_by_hand_example_matches_the_chapter():
    m = feature()
    assert cost_per_attempt(m) == D("0.63")
    assert cost_per_success(m) == D("0.7875")
    assert margin(m).contribution == D("0.7125")


def test_a_failed_attempt_is_still_paid_for():
    assert cost_per_success(feature("0.4")) == 2 * cost_per_success(
        feature("0.8")
    )


def test_the_stack_adds_up_to_the_total():
    m = feature()
    assert sum(cost_stack(m).values()) == cost_per_success(m)


def test_token_cost_uses_call_cost():
    haiku = load_catalog().get("claude-haiku-5-5")
    calls = (Usage(input_tokens=10_000, output_tokens=1_000),) * 3
    line = token_component(haiku, calls)
    assert line.per_attempt == D("0.004500")
    batched = token_component(haiku, calls, batch=True)
    assert batched.per_attempt == line.per_attempt / 2


def test_break_even_price_and_volume():
    m = feature()
    assert break_even_price(m) == D("0.7875")
    assert break_even_volume(m, D("7125")) == 10_000
    with pytest.raises(ValueError, match="non-positive"):
        break_even_volume(feature(price="0.50"), D(100))


def test_success_rate_must_be_a_probability():
    with pytest.raises(ValueError, match="success_rate"):
        feature(rate="0")
    with pytest.raises(ValueError, match="success_rate"):
        feature(rate="1.2")


def test_sensitivity_orders_by_swing_and_clips_success_at_one():
    rows = sensitivity(
        feature(),
        {"success_rate": D("0.1"), "tokens": D("0.5"), "price": D("0.1")},
    )
    assert [r.name for r in rows] == ["price", "success_rate", "tokens"]
    top = rows[1]
    assert top.high == margin(feature("0.88")).contribution
    assert sensitivity(feature("1"), {"success_rate": D("0.2")})[0].high == (
        sensitivity(feature("1"), {"success_rate": D(0)})[0].high
    )
    with pytest.raises(KeyError):
        sensitivity(feature(), {"weather": D("0.1")})


def test_crossover_needs_a_saving():
    assert crossover_volume(D("0.05"), D(3000), D("0.02")) == 100_000
    assert crossover_volume(D("0.02"), D(3000), D("0.02")) is None


def test_budget_evidence_gates_on_cost():
    ok = budget_evidence(
        "gen", feature(), D("0.80"), run_id="m1", recorded_at=0
    )
    bad = budget_evidence(
        "gen", feature(), D("0.70"), run_id="m1", recorded_at=0
    )
    assert (ok.verdict, bad.verdict) == ("pass", "fail")
