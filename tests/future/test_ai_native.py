from __future__ import annotations

import random

import pytest

from ai_engineer.future.ai_native import (
    Criterion,
    Spec,
    Stage,
    amdahl_speedup,
    check_spec,
    lead_time,
    simulate_review,
)

STAGES = (
    Stage("idea", 2),
    Stage("spec", 1),
    Stage("code", 3),
    Stage("review", 2),
    Stage("test", 1),
    Stage("release", 1),
)


def test_amdahl_matches_hand_value() -> None:
    got = amdahl_speedup([0.3, 0.7], [10, 1])
    assert got == pytest.approx(1 / (0.03 + 0.7))
    assert amdahl_speedup([1.0], [4]) == pytest.approx(4.0)


def test_amdahl_is_capped_by_the_rest() -> None:
    assert amdahl_speedup([0.3, 0.7], [1e9, 1]) < 1 / 0.7 + 1e-6


@pytest.mark.parametrize(
    ("fractions", "speedups"),
    [([0.5], [1.0]), ([0.5, 0.5], [1.0]), ([1.0], [0.0]), ([], [])],
)
def test_amdahl_rejects_bad_input(
    fractions: list[float], speedups: list[float]
) -> None:
    with pytest.raises(ValueError, match="must"):
        amdahl_speedup(fractions, speedups)


def test_lead_time_by_hand() -> None:
    assert lead_time(STAGES) == 10
    fast = [s.scaled(10) if s.name == "code" else s for s in STAGES]
    assert lead_time(fast) == pytest.approx(7.3)
    quick = [s.scaled(2) if s.name == "review" else s for s in STAGES]
    assert lead_time(quick) == 9
    with pytest.raises(ValueError, match="positive"):
        STAGES[0].scaled(0)


def test_check_spec_reports_missing_and_orphans() -> None:
    spec = Spec(
        "refund",
        (
            Criterion("AC1", "refund within 30 days", "test_refund_ok"),
            Criterion("AC2", "no refund after 30 days", "test_refund_late"),
        ),
    )
    ok = check_spec(spec, ["test_refund_ok", "test_refund_late"])
    assert ok.ok
    bad = check_spec(spec, ["test_refund_ok", "test_extra"])
    assert bad.missing == ("AC2",)
    assert bad.orphans == ("test_extra",)
    assert not bad.ok


def test_review_is_deterministic_and_stable_when_fast() -> None:
    a = simulate_review(10, 6, 2.0, 60, random.Random(2030))
    b = simulate_review(10, 6, 2.0, 60, random.Random(2030))
    assert a == b
    assert a.queue_length < 10
    assert a.p50_hours < a.p90_hours <= a.p99_hours


def test_overload_grows_the_queue(rng: random.Random) -> None:
    calm = simulate_review(20, 6, 2.0, 60, random.Random(2030))
    flood = simulate_review(40, 6, 2.0, 60, rng)
    assert flood.queue_length > 10 * max(calm.queue_length, 1)
    assert flood.p90_hours > 5 * calm.p90_hours


def test_small_prs_and_precheck_restore_flow() -> None:
    flood = simulate_review(40, 6, 2.0, 60, random.Random(2030))
    fixed = simulate_review(30, 6, 1.0, 60, random.Random(2030))
    assert fixed.queue_length < flood.queue_length / 10
    assert fixed.p90_hours < flood.p90_hours / 5


def test_review_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="positive"):
        simulate_review(0, 6, 2.0, 10, random.Random(1))
