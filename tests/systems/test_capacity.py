"""Tests for the capacity arithmetic of chapter 54."""

from __future__ import annotations

import random
from decimal import Decimal

import pytest

from ai_engineer.systems.capacity import (
    Estimate,
    LatencyBudget,
    fanout_slow_share,
    littles_law,
    mm1_wait,
    simulate_fanout,
)


def case() -> Estimate:
    return Estimate(2_000_000, 5, 6.0, 3_000, 400)


# region test-estimate
def test_estimate_by_hand_case() -> None:
    e = case()
    assert e.daily_requests == 10_000_000
    assert e.average_rps == pytest.approx(115.74, abs=0.01)
    assert e.peak_rps == pytest.approx(694.44, abs=0.01)
    tin, tout = e.peak_tokens_per_second()
    assert tin == pytest.approx(2_083_333, abs=1)
    assert tout == pytest.approx(277_778, abs=1)
    assert e.monthly_cost(Decimal(2), Decimal(10)) == Decimal("3000000")


# endregion test-estimate


def test_littles_law_and_mm1() -> None:
    assert littles_law(50, 2.0) == 100
    assert mm1_wait(0.5, 1.0) == pytest.approx(1.0)
    assert mm1_wait(0.9, 1.0) == pytest.approx(9.0)
    with pytest.raises(ValueError, match="must"):
        mm1_wait(1.0, 1.0)
    with pytest.raises(ValueError, match="must"):
        littles_law(-1, 1)


# region test-budget
def test_budget_lists_hops_over_allowance() -> None:
    b = LatencyBudget((("gateway", 0.1), ("retrieve", 0.4), ("generate", 1.0)))
    assert b.total == pytest.approx(1.5)
    over = b.check({"gateway": 0.05, "retrieve": 0.6, "generate": 1.0})
    assert over == ("retrieve",)


# endregion test-budget


def test_fanout_share() -> None:
    assert fanout_slow_share(0.01, 1) == pytest.approx(0.01)
    assert fanout_slow_share(0.01, 100) == pytest.approx(0.634, abs=0.001)


# region test-hedge
def test_hedging_cuts_p99_and_costs_a_little_load() -> None:
    plain = simulate_fanout(random.Random(2030), 100)
    hedged = simulate_fanout(random.Random(2030), 100, hedge_after=0.25)
    assert plain.extra_load == 0.0
    assert hedged.p99 < plain.p99
    assert 0 < hedged.extra_load < 0.08


# endregion test-hedge


def test_fanout_p99_grows_with_n() -> None:
    p = [simulate_fanout(random.Random(2030), n).p99 for n in (1, 10, 100)]
    assert p[0] < p[1] < p[2]
