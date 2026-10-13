from __future__ import annotations

import numpy as np
import pytest

from ai_engineer.future.simulation import (
    Floats,
    Surrogate,
    active_learning,
    bootstrap_ensemble,
    ensemble_disagreement,
    estimate,
    monte_carlo,
    queue_wait,
    random_sampling,
    run_until,
)


def inside_circle(rng: np.random.Generator, n: int) -> Floats:
    pts = rng.random((n, 2))
    hit = np.sum(pts**2, axis=1) <= 1.0
    return np.asarray(4.0 * hit, dtype=np.float64)


def test_monte_carlo_interval_covers_pi() -> None:
    est = monte_carlo(inside_circle, 20_000, np.random.default_rng(2030))
    assert est.low < np.pi < est.high
    assert est.n == 20_000


def test_error_shrinks_as_root_n() -> None:
    rng = np.random.default_rng(2030)
    small = monte_carlo(inside_circle, 1_000, rng).half_width
    big = monte_carlo(inside_circle, 16_000, rng).half_width
    assert small / big == pytest.approx(4.0, rel=0.15)


def test_estimate_needs_two_draws() -> None:
    with pytest.raises(ValueError, match="two draws"):
        estimate(np.array([1.0]))


def test_run_until_stops_at_half_width() -> None:
    rng = np.random.default_rng(2030)
    est = run_until(inside_circle, rng, 0.05)
    assert est.half_width <= 0.05
    assert est.n < 1_000_000
    capped = run_until(inside_circle, rng, 1e-9, max_n=3_000)
    assert capped.n == 3_000


def test_surrogate_exact_on_a_cubic_and_flags_range() -> None:
    x = np.linspace(0.0, 1.0, 9)
    s = Surrogate.fit(x, x**3 - x, degree=3)
    assert s.predict(x) == pytest.approx(x**3 - x)
    assert list(s.inside(np.array([0.5, 1.5]))) == [True, False]


def test_surrogate_fails_outside_its_range() -> None:
    train = np.linspace(0.2, 0.8, 7)
    s = Surrogate.fit(train, np.array([queue_wait(r) for r in train]))
    truth = np.array([queue_wait(r) for r in (0.5, 0.95)])
    err = np.abs(s.predict(np.array([0.5, 0.95])) - truth)
    assert err[1] > 5 * err[0]


def test_ensemble_disagrees_more_far_from_data() -> None:
    rng = np.random.default_rng(2030)
    x = np.linspace(0.0, 1.0, 12)
    y = np.sin(3 * x) + rng.normal(0, 0.05, 12)
    models = bootstrap_ensemble(x, y, rng, size=20)
    near = ensemble_disagreement(models, np.array([0.5]))[0]
    far = ensemble_disagreement(models, np.array([1.0]))[0]
    assert far > near


def test_active_learning_spends_exactly_the_budget() -> None:
    cand = np.linspace(0.1, 0.9, 41)
    x, y = active_learning(
        lambda v: v**2, cand, 9, np.random.default_rng(2030)
    )
    assert len(x) == len(y) == 9
    assert len(set(x.tolist())) == 9
    rx, _ = random_sampling(lambda v: v**2, cand, 9, np.random.default_rng(1))
    assert len(rx) == 9


def test_active_learning_rejects_bad_budget() -> None:
    with pytest.raises(ValueError, match="budget"):
        active_learning(
            lambda v: v, np.linspace(0, 1, 10), 3, np.random.default_rng(2030)
        )


def test_queue_wait_is_repeatable_and_grows() -> None:
    assert queue_wait(0.5) == queue_wait(0.5)
    assert queue_wait(0.8) > queue_wait(0.5)
