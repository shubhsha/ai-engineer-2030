"""Human-AI collaboration: deferral, calibration and automation bias.

A *case* is one decision the model could make alone. We record how sure
the model was, whether it was right, and whether a person working alone
would have been right. From those three facts we can price any policy that
sends some cases to the person. Everything here is offline arithmetic.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Case:
    """One decision: the model's confidence and both parties' accuracy."""

    confidence: float
    model_correct: bool
    human_correct: bool


@dataclass(frozen=True, slots=True)
class Outcome:
    """What a policy achieved on a set of cases."""

    accuracy: float
    coverage: float
    reviewed: int
    errors: int


@dataclass(frozen=True, slots=True)
class Bin:
    """One reliability bin: how sure the model said it was, how right."""

    low: float
    high: float
    count: int
    mean_confidence: float
    accuracy: float


def make_cases(
    n: int,
    rng: random.Random,
    *,
    human_accuracy: float = 0.8,
    overconfidence: float = 0.0,
) -> list[Case]:
    """Draw synthetic cases; the person is right with fixed probability.

    The model's true chance of being right is its stated confidence minus
    ``overconfidence``, so 0.0 gives a calibrated model.
    """
    cases: list[Case] = []
    for _ in range(n):
        conf = min(0.999, max(0.3, rng.betavariate(5.0, 1.6)))
        true_p = min(1.0, max(0.0, conf - overconfidence))
        cases.append(
            Case(
                confidence=conf,
                model_correct=rng.random() < true_p,
                human_correct=rng.random() < human_accuracy,
            )
        )
    return cases


# region team_outcome
def team_outcome(cases: Sequence[Case], threshold: float) -> Outcome:
    """Model answers when confidence >= threshold; a person decides rest.

    ``coverage`` is the share of cases the model settles without a person.
    """
    if not cases:
        raise ValueError("no cases")
    right = 0
    reviewed = 0
    for c in cases:
        if c.confidence >= threshold:
            right += c.model_correct
        else:
            reviewed += 1
            right += c.human_correct
    n = len(cases)
    return Outcome(
        accuracy=right / n,
        coverage=(n - reviewed) / n,
        reviewed=reviewed,
        errors=n - right,
    )


# endregion


def sweep(
    cases: Sequence[Case], thresholds: Sequence[float]
) -> list[tuple[float, Outcome]]:
    """Team outcome at each threshold; 0.0 is model alone, above 1 is all."""
    return [(t, team_outcome(cases, t)) for t in thresholds]


def model_accuracy(cases: Sequence[Case]) -> float:
    return team_outcome(cases, 0.0).accuracy


def human_accuracy(cases: Sequence[Case]) -> float:
    return team_outcome(cases, math.inf).accuracy


# region reliability
def reliability_bins(cases: Sequence[Case], bins: int = 10) -> list[Bin]:
    """Group cases by stated confidence; report the real hit rate in each."""
    if bins < 1:
        raise ValueError("bins must be at least 1")
    groups: list[list[Case]] = [[] for _ in range(bins)]
    for c in cases:
        groups[min(bins - 1, int(c.confidence * bins))].append(c)
    out: list[Bin] = []
    for i, g in enumerate(groups):
        if not g:
            continue
        out.append(
            Bin(
                low=i / bins,
                high=(i + 1) / bins,
                count=len(g),
                mean_confidence=sum(c.confidence for c in g) / len(g),
                accuracy=sum(c.model_correct for c in g) / len(g),
            )
        )
    return out


def ece(cases: Sequence[Case], bins: int = 10) -> float:
    """Expected calibration error: bin-weighted gap of confidence to hits."""
    if not cases:
        raise ValueError("no cases")
    n = len(cases)
    return sum(
        b.count / n * abs(b.accuracy - b.mean_confidence)
        for b in reliability_bins(cases, bins)
    )


# endregion


# region simulate_bias
def simulate_bias(
    cases: Sequence[Case],
    accept_prob: float,
    rng: random.Random,
    *,
    threshold: float = math.inf,
) -> Outcome:
    """Reviewer sees the model's answer on cases below ``threshold``.

    With probability ``accept_prob`` they accept it without checking (so
    the case is right when the model is). Otherwise they decide alone.
    """
    if not 0.0 <= accept_prob <= 1.0:
        raise ValueError("accept_prob must be in [0, 1]")
    right = 0
    reviewed = 0
    for c in cases:
        if c.confidence >= threshold:
            right += c.model_correct
        elif rng.random() < accept_prob:
            reviewed += 1
            right += c.model_correct
        else:
            reviewed += 1
            right += c.human_correct
    n = len(cases)
    return Outcome(
        accuracy=right / n,
        coverage=(n - reviewed) / n,
        reviewed=reviewed,
        errors=n - right,
    )


# endregion
