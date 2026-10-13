"""Tests for the engineering economics module."""

from __future__ import annotations

import random
from decimal import Decimal

from ai_engineer.claude.catalog import load_catalog
from ai_engineer.leadership.economics import (
    Job,
    Tco,
    breakeven_changes,
    build_vs_buy,
    debt_interest,
    model_spend_per_year,
    monthly_rate,
    npv,
    order_cd3,
    order_fifo,
    order_sjf,
    payback_month,
    total_delay_cost,
)
from ai_engineer.llm.model import Usage

D = Decimal


def project() -> list[Decimal]:
    return [D(-60000)] + [D(3000)] * 24


def test_monthly_rate_compounds_back() -> None:
    r = monthly_rate(D("0.10"))
    assert abs((1 + r) ** 12 - D("1.10")) < D("1e-20")


def test_npv_zero_rate_is_sum() -> None:
    assert npv(D(0), project()) == D(12000)


def test_npv_discounts() -> None:
    assert npv(D("0.10"), [D(-100), D(110)]) == D("0.00")
    assert npv(monthly_rate(D("0.10")), project()) < D(12000)


def test_payback_month() -> None:
    assert payback_month(project()) == 20
    assert payback_month([D(-10), D(1)]) is None


def test_build_vs_buy_exit_cost_flips_verdict() -> None:
    build = Tco("build", D(90000), D(40000))
    buy = Tco("buy", D(20000), D(55000))
    assert build_vs_buy(build, buy, 3).cheaper == "buy"
    exit_ = Tco("buy", D(20000), D(55000), D(120000))
    verdict = build_vs_buy(build, exit_, 3)
    assert verdict.cheaper == "build"
    assert verdict.gap == D(95000)


def test_model_spend_uses_call_cost() -> None:
    model = load_catalog().get("claude-sonnet-5-5")
    usage = Usage(input_tokens=1000, output_tokens=200)
    assert model_spend_per_year(model, usage, 1000) == D("4")


def backlog(rng: random.Random) -> list[Job]:
    return [
        Job(
            f"J{i}",
            D(rng.randrange(1, 20)) * 1000,
            D(rng.randrange(1, 9)),
        )
        for i in range(12)
    ]


def test_cd3_beats_fifo_and_sjf(rng: random.Random) -> None:
    jobs = backlog(rng)
    fifo = total_delay_cost(order_fifo(jobs))
    sjf = total_delay_cost(order_sjf(jobs))
    cd3 = total_delay_cost(order_cd3(jobs))
    assert cd3 <= min(fifo, sjf)


def test_equal_durations_sjf_is_arbitrary() -> None:
    a = Job("a", D(1000), D(2))
    b = Job("b", D(9000), D(2))
    assert total_delay_cost([a, b]) == D(2000 + 9000 * 4)
    assert total_delay_cost(order_cd3([a, b])) == D(9000 * 2 + 1000 * 4)


def test_debt_interest_stops_at_paydown() -> None:
    assert debt_interest(10, D(3)) == D(30)
    assert debt_interest(10, D(3), paid_down_at=4) == D(12)
    assert debt_interest(2, D(3), paid_down_at=4) == D(6)


def test_breakeven_changes_rounds_up() -> None:
    assert breakeven_changes(D(40), D(3)) == 14
    assert breakeven_changes(D(42), D(3)) == 14
