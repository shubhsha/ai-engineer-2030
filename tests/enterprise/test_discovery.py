from __future__ import annotations

import random

import pytest

from ai_engineer.enterprise.discovery import (
    Component,
    ProblemStatement,
    Threshold,
    UseCase,
    Weights,
    handoff_gaps,
    kendall_tau,
    rank_stability,
    rank_use_cases,
    readiness,
)


def test_problem_statement_needs_a_number_that_moves() -> None:
    ps = ProblemStatement(
        "Release testers",
        "spend two days on regression",
        "days to sign",
        2,
        0.5,
        90,
    )
    assert "from 2 to 0.5 within 90 days" in ps.text()
    with pytest.raises(ValueError, match="empty"):
        ProblemStatement("a", "b", "", 1, 2, 30)
    with pytest.raises(ValueError, match="differ"):
        ProblemStatement("a", "b", "m", 1, 1, 30)
    with pytest.raises(ValueError, match="positive"):
        ProblemStatement("a", "b", "m", 1, 2, 0)


GOOD = {
    "data_access": 0.9,
    "data_quality": 0.8,
    "labeled_examples": 0.7,
    "owner": 1.0,
    "risk_tier": 0.8,
}


def test_readiness_go_fix_stop() -> None:
    assert readiness(GOOD).verdict == "go"
    fix = readiness({**GOOD, "data_quality": 0.5})
    assert fix.verdict == "fix"
    assert "data_quality" in fix.reasons[0]
    stop = readiness({**GOOD, "data_quality": 0.5, "labeled_examples": 0.1})
    assert stop.verdict == "stop"
    assert len(stop.reasons) == 2


def test_missing_dimension_is_a_stop() -> None:
    scores = {k: v for k, v in GOOD.items() if k != "owner"}
    r = readiness(scores)
    assert r.verdict == "stop"
    assert r.reasons == ("owner: not assessed",)


def test_threshold_validated() -> None:
    with pytest.raises(ValueError, match="stop <= go"):
        Threshold(0.3, 0.6)


CASES = [
    UseCase("big-risky", 9, 5, 8),
    UseCase("small-safe", 4, 9, 1),
    UseCase("middle", 6, 6, 4),
]


def test_ranking_follows_weights() -> None:
    assert rank_use_cases(CASES, Weights(1, 0, 0))[0] == "big-risky"
    assert rank_use_cases(CASES, Weights(0, 0, 1))[0] == "small-safe"
    assert rank_use_cases(CASES, Weights(1, 1, 1)) == [
        "small-safe",
        "middle",
        "big-risky",
    ]


def test_weights_validated() -> None:
    with pytest.raises(ValueError, match="zero"):
        Weights(0, 0, 0)
    with pytest.raises(ValueError, match="negative"):
        Weights(-1, 1, 1)


def test_kendall_tau_extremes() -> None:
    assert kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert kendall_tau(["a", "b", "c"], ["c", "b", "a"]) == -1.0


def test_stability_identical_and_different() -> None:
    same = rank_stability(CASES, [Weights(1, 1, 1), Weights(2, 2, 2)])
    assert same.mean_tau == 1.0
    assert same.top_share == 1.0
    diff = rank_stability(CASES, [Weights(1, 0, 0), Weights(0, 0, 1)])
    assert diff.mean_tau < 1.0
    assert diff.top_share == 0.5


def test_seeded_candidates_are_reproducible(rng: random.Random) -> None:
    def make(r: random.Random) -> list[UseCase]:
        return [
            UseCase(
                f"uc{i}", r.uniform(0, 10), r.uniform(0, 10), r.uniform(0, 10)
            )
            for i in range(10)
        ]

    a = rank_use_cases(make(rng), Weights(1, 1, 1))
    b = rank_use_cases(make(random.Random(2030)), Weights(1, 1, 1))
    assert a == b


def test_handoff_gaps_lists_what_is_missing() -> None:
    comps = [
        Component("api", "ana", "rb.md", "99.5%", "ev/1", "$400"),
        Component("index", "ben", None, "  ", "ev/2", None),
        Component("nightly"),
    ]
    gaps = handoff_gaps(comps)
    assert "api" not in gaps
    assert gaps["index"] == ("runbook", "slo", "budget")
    assert gaps["nightly"] == (
        "owner",
        "runbook",
        "slo",
        "evidence",
        "budget",
    )
