"""Tests for deferral, calibration and automation-bias arithmetic."""

from __future__ import annotations

import random

import pytest

from ai_engineer.future.collaboration import (
    Case,
    ece,
    human_accuracy,
    make_cases,
    model_accuracy,
    reliability_bins,
    simulate_bias,
    sweep,
    team_outcome,
)

CASES = [
    Case(0.95, True, False),
    Case(0.90, True, True),
    Case(0.60, False, True),
    Case(0.50, False, True),
]


def test_threshold_zero_is_model_alone_and_inf_is_person_alone() -> None:
    assert model_accuracy(CASES) == 0.5
    assert human_accuracy(CASES) == 0.75


def test_team_beats_both_when_model_knows_what_it_does_not_know() -> None:
    out = team_outcome(CASES, 0.7)
    assert out.accuracy == 1.0
    assert out.coverage == 0.5
    assert out.reviewed == 2
    assert out.errors == 0


def test_sweep_matches_team_outcome() -> None:
    rows = sweep(CASES, [0.0, 0.7])
    assert [t for t, _ in rows] == [0.0, 0.7]
    assert rows[1][1] == team_outcome(CASES, 0.7)


def test_empty_cases_rejected() -> None:
    with pytest.raises(ValueError, match="no cases"):
        team_outcome([], 0.5)
    with pytest.raises(ValueError, match="no cases"):
        ece([])


def test_bins_count_every_case_once() -> None:
    bins = reliability_bins(CASES, 5)
    assert sum(b.count for b in bins) == len(CASES)
    assert reliability_bins(CASES, 5)[-1].high == 1.0
    with pytest.raises(ValueError, match="bins"):
        reliability_bins(CASES, 0)


def test_ece_by_hand() -> None:
    cases = [Case(0.9, True, True)] * 5 + [Case(0.9, False, True)] * 5
    assert ece(cases, 10) == pytest.approx(0.4)


def test_calibrated_model_has_low_ece_overconfident_high(
    rng: random.Random,
) -> None:
    good = ece(make_cases(5000, rng))
    bad = ece(make_cases(5000, rng, overconfidence=0.2))
    assert good < 0.03
    assert bad > 0.15


def test_deferral_beats_both_alone(rng: random.Random) -> None:
    cases = make_cases(5000, rng)
    team = team_outcome(cases, 0.7).accuracy
    assert team > model_accuracy(cases)
    assert team > human_accuracy(cases)


def test_bias_erases_gain_monotonically(rng: random.Random) -> None:
    cases = make_cases(5000, rng)
    accs = [
        simulate_bias(cases, p, random.Random(2030), threshold=0.7).accuracy
        for p in (0.0, 0.5, 1.0)
    ]
    assert accs[0] > accs[1] > accs[2]
    assert accs[2] == team_outcome(cases, 0.0).accuracy


def test_bias_zero_with_all_review_is_person_alone() -> None:
    out = simulate_bias(CASES, 0.0, random.Random(1))
    assert out.accuracy == 0.75
    assert out.coverage == 0.0


def test_bad_accept_prob_rejected() -> None:
    with pytest.raises(ValueError, match="accept_prob"):
        simulate_bias(CASES, 1.5, random.Random(1))
