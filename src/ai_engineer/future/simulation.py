"""Monte Carlo, surrogate models and active learning (NumPy only)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Floats = NDArray[np.float64]
Draws = Callable[[np.random.Generator, int], Floats]
Simulator = Callable[[float], float]

Z95 = 1.96


# region estimate
@dataclass(frozen=True, slots=True)
class Estimate:
    mean: float
    half_width: float
    n: int

    @property
    def low(self) -> float:
        return self.mean - self.half_width

    @property
    def high(self) -> float:
        return self.mean + self.half_width


def estimate(draws: Floats) -> Estimate:
    """Mean and 95% half-width of independent draws."""
    n = len(draws)
    if n < 2:
        raise ValueError("need at least two draws")
    se = float(np.std(draws, ddof=1)) / np.sqrt(n)
    return Estimate(float(np.mean(draws)), Z95 * se, n)


def monte_carlo(f: Draws, n: int, rng: np.random.Generator) -> Estimate:
    """Average n independent draws of f and report the 95% interval."""
    return estimate(f(rng, n))


# endregion estimate


# region run-until
def run_until(
    f: Draws,
    rng: np.random.Generator,
    half_width: float,
    batch: int = 1_000,
    max_n: int = 1_000_000,
) -> Estimate:
    """Draw in batches until the interval is narrow enough (or max_n)."""
    draws = f(rng, batch)
    while len(draws) < max_n:
        if estimate(draws).half_width <= half_width:
            break
        draws = np.concatenate([draws, f(rng, batch)])
    return estimate(draws)


# endregion run-until


# region surrogate
@dataclass(frozen=True, slots=True)
class Surrogate:
    """A polynomial fitted to simulator runs on one input."""

    coef: Floats
    lo: float
    hi: float

    @classmethod
    def fit(cls, x: Floats, y: Floats, degree: int = 3) -> Surrogate:
        lo, hi = float(np.min(x)), float(np.max(x))
        u = _scale(x, lo, hi)
        coef = np.asarray(
            np.polynomial.polynomial.polyfit(u, y, degree), dtype=np.float64
        )
        return cls(coef, lo, hi)

    def predict(self, x: Floats) -> Floats:
        u = _scale(x, self.lo, self.hi)
        return np.asarray(
            np.polynomial.polynomial.polyval(u, self.coef), dtype=np.float64
        )

    def inside(self, x: Floats) -> NDArray[np.bool_]:
        return (x >= self.lo) & (x <= self.hi)


def _scale(x: Floats, lo: float, hi: float) -> Floats:
    return (x - lo) / (hi - lo) * 2.0 - 1.0


# endregion surrogate


# region ensemble
def bootstrap_ensemble(
    x: Floats,
    y: Floats,
    rng: np.random.Generator,
    size: int = 12,
    degree: int = 3,
) -> list[Surrogate]:
    """Fit surrogates on resamples (with replacement) of the runs."""
    models: list[Surrogate] = []
    while len(models) < size:
        idx = rng.integers(0, len(x), len(x))
        if len(np.unique(x[idx])) <= degree:
            continue  # too few distinct points to fit this degree
        models.append(Surrogate.fit(x[idx], y[idx], degree))
    return models


def ensemble_disagreement(models: list[Surrogate], x: Floats) -> Floats:
    """Standard deviation of the models' predictions at each x."""
    spread = np.std([m.predict(x) for m in models], axis=0, ddof=1)
    return np.asarray(spread, dtype=np.float64)


# endregion ensemble


# region active
def active_learning(
    sim: Simulator,
    candidates: Floats,
    budget: int,
    rng: np.random.Generator,
    start: int = 5,
    degree: int = 3,
) -> tuple[Floats, Floats]:
    """Spend the run budget where the ensemble disagrees most."""
    if budget < start or budget > len(candidates):
        raise ValueError("budget must fit between start and candidates")
    if start <= degree:
        raise ValueError("start must exceed the polynomial degree")
    pick = [int(i) for i in rng.choice(len(candidates), start, replace=False)]
    runs = [sim(float(candidates[i])) for i in pick]
    while len(pick) < budget:
        models = bootstrap_ensemble(
            candidates[pick], np.array(runs), rng, degree=degree
        )
        spread = ensemble_disagreement(models, candidates)
        spread[pick] = -1.0  # never repeat a run
        pick.append(int(np.argmax(spread)))
        runs.append(sim(float(candidates[pick[-1]])))
    return candidates[pick], np.array(runs)


def random_sampling(
    sim: Simulator,
    candidates: Floats,
    budget: int,
    rng: np.random.Generator,
) -> tuple[Floats, Floats]:
    """The baseline: the same budget spent on random candidates."""
    pick = rng.choice(len(candidates), budget, replace=False)
    x = candidates[pick]
    return x, np.array([sim(float(v)) for v in x])


# endregion active


# region queue-sim
def queue_wait(rho: float, jobs: int = 4_000, seed: int = 7) -> float:
    """Mean wait in a one-server queue; service time 1, load rho.

    Lindley's recursion: wait[i+1] = max(0, wait[i] + service - gap).
    The fixed seed makes each run repeatable for a given rho.
    """
    rng = np.random.default_rng(seed)
    gap = rng.exponential(1.0 / rho, jobs)
    service = rng.exponential(1.0, jobs)
    wait = 0.0
    total = 0.0
    for g, s in zip(gap, service, strict=True):
        wait = max(0.0, wait + s - g)
        total += wait
    return total / jobs


# endregion queue-sim
