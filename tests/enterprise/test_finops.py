"""Tests for the cost ledger, the waterfall and the budget alarm."""

from __future__ import annotations

import random
from decimal import Decimal

import pytest

from ai_engineer.claude.catalog import Catalog, load_catalog
from ai_engineer.claude.cost import call_cost
from ai_engineer.enterprise import finops as f
from ai_engineer.llm.model import Usage


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    return load_catalog()


@pytest.fixture
def month(catalog: Catalog, rng: random.Random) -> f.Ledger:
    return f.ledger_from_calls(f.synthetic_month(rng), catalog)


def _pairs(n: int, a_only: int, b_only: int) -> tuple[list[bool], list[bool]]:
    a, b = [True] * n, [True] * n
    for i in range(a_only):
        b[i] = False
    for i in range(b_only):
        a[n - 1 - i] = False
    return a, b


def _feature(name: str) -> f.Match:
    return lambda e: e.feature == name


def test_ledger_prices_each_call_with_call_cost(catalog: Catalog) -> None:
    usage = Usage(input_tokens=1000, output_tokens=100)
    call = f.Call("t", "x", "prod", "claude-sonnet-5-5", usage)
    (entry,) = f.ledger_from_calls([call], catalog)
    expected = call_cost(catalog.get("claude-sonnet-5-5"), usage)
    assert entry.cost == expected == Decimal("0.003")


def test_allocation_sums_to_the_total(month: f.Ledger) -> None:
    for by in ("team", "feature", "env", "model"):
        assert sum(f.allocate(month, by).values()) == f.total(month)
    with pytest.raises(ValueError, match=r"cannot"):
        f.allocate(month, "color")


def test_untagged_spend_is_grouped_and_measured(month: f.Ledger) -> None:
    share = f.untagged_share(month)
    assert Decimal("0.03") < share < Decimal("0.08")
    assert f.UNTAGGED in f.allocate(month, "feature")
    assert f.untagged_share(()) == 0


def test_seeded_month_is_deterministic(catalog: Catalog) -> None:
    a = f.synthetic_month(random.Random(2030))
    b = f.synthetic_month(random.Random(2030))
    assert a == b


def test_reconcile_passes_inside_tolerance_and_fails_outside(
    month: f.Ledger,
) -> None:
    whole = f.total(month)
    ok = f.reconcile(
        month, whole * Decimal("1.005"), run_id="r", recorded_at=0
    )
    bad = f.reconcile(
        month, whole * Decimal("1.05"), run_id="r", recorded_at=0
    )
    assert (ok.verdict, bad.verdict) == ("pass", "fail")


def test_savings_multiply_they_do_not_add(
    month: f.Ledger, catalog: Catalog
) -> None:
    ok = f.paired_guard(*_pairs(100, 0, 0), "x")
    batch = f.to_batch(_feature("doc-summary"), catalog)
    trim = f.trim_input(_feature("doc-summary"), Decimal("0.5"), catalog)
    levers = [f.Lever("trim", trim, ok), f.Lever("batch", batch, ok)]
    one, two = f.waterfall(month, levers)
    assert two.before == one.after
    alone = f.waterfall(month, [f.Lever("batch", batch, ok)])[0]
    # Batch saves less after the trim than it would have alone, because
    # the trim already removed part of what the discount would halve.
    assert two.saving < alone.saving
    assert one.saving + alone.saving > f.total(month) - two.after


def test_a_failing_guard_blocks_the_lever(
    month: f.Ledger, catalog: Catalog
) -> None:
    bad = f.paired_guard(*_pairs(200, 14, 3), "route")
    lever = f.Lever(
        "route",
        f.route(_feature("sql-agent"), "claude-sonnet-5-5", catalog),
        bad,
    )
    (step,) = f.waterfall(month, [lever])
    assert not step.kept
    assert step.saving == 0
    assert step.guard.verdict == "fail"


def test_cache_prefix_moves_tokens_and_cuts_cost(
    month: f.Ledger, catalog: Catalog
) -> None:
    after = f.cache_prefix(
        _feature("support-chat"), 2400, Decimal("0.9"), catalog
    )(month)
    assert f.total(after) < f.total(month)
    moved = next(e for e in after if e.feature == "support-chat")
    assert moved.usage.cache_read_tokens == 2160
    assert moved.usage.cache_write_tokens == 240


def test_route_to_haiku_5_5_reprices_the_call(catalog: Catalog) -> None:
    usage = Usage(input_tokens=800, output_tokens=20)
    (entry,) = f.ledger_from_calls(
        [f.Call("t", "x", "prod", "claude-sonnet-5-5", usage)], catalog
    )
    (moved,) = f.route(_feature("x"), "claude-haiku-5-5", catalog)((entry,))
    assert moved.model == "claude-haiku-5-5"
    assert moved.cost == entry.cost / 20


def test_drop_removes_waste(month: f.Ledger) -> None:
    kept = f.drop(_feature("legacy-digest"))(month)
    assert all(e.feature != "legacy-digest" for e in kept)
    assert len(kept) < len(month)


def test_long_prompt_price_step_is_about_five_times(
    catalog: Catalog,
) -> None:
    haiku = catalog.get("claude-haiku-5-5")
    short = call_cost(haiku, Usage(input_tokens=99_000, output_tokens=1000))
    long = call_cost(haiku, Usage(input_tokens=101_000, output_tokens=1000))
    assert short == Decimal("0.0104")
    assert long == Decimal("0.053")
    assert 5 < long / short < 5.2


def test_budget_alerts_flag_over_and_fast_burn(month: f.Ledger) -> None:
    budgets = {"sql-agent": Decimal(60), "doc-summary": Decimal(120)}
    assert f.budget_alerts(month, budgets, 1.0) == ()
    (alert,) = f.budget_alerts(month, {"sql-agent": Decimal(60)}, 0.3)
    assert alert.level == "burn"
    assert alert.burn > 2
    (over,) = f.budget_alerts(month, {"sql-agent": Decimal(40)}, 1.0)
    assert over.level == "over"
    with pytest.raises(ValueError, match=r"in \(0, 1\]"):
        f.budget_alerts(month, budgets, 0)


def test_retry_loop_trips_budget_and_anomaly(
    month: f.Ledger, catalog: Catalog
) -> None:
    usage = Usage(input_tokens=4000, output_tokens=800)
    loop = f.ledger_from_calls(
        [
            f.Call("data", "sql-agent", "prod", "claude-opus-5-5", usage, 29)
            for _ in range(3000)
        ],
        catalog,
    )
    assert f.spend_anomaly(month, 29, 4).flagged is False
    assert f.spend_anomaly(month + loop, 29, 4).flagged is True
    alerts = f.budget_alerts(month + loop, {"sql-agent": Decimal(60)}, 1.0)
    assert alerts[0].level == "over"


def test_self_host_break_even_in_tokens() -> None:
    tokens = f.self_host_break_even(Decimal(12000), Decimal("3.5"))
    assert tokens == 3_428_571_428
