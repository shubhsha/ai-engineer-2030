"""Chapter 61: a decision matrix whose result you can stress-test.

Options are scored on weighted criteria. The total of an option is the
sum of weight times score. ``flip_margin`` reports, for each criterion,
the smallest change in that one weight (others held fixed) at which a
different option ties for the lead, so a group can see how robust the
choice is. ``order_problems`` checks that the criteria were recorded
before any score was.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class Option:
    name: str


@dataclass(frozen=True, slots=True)
class Criterion:
    name: str
    weight: Decimal


@dataclass(frozen=True, slots=True)
class Ranked:
    option: str
    total: Decimal


Scores = Mapping[tuple[str, str], int]


def _total(
    option: Option, criteria: Sequence[Criterion], scores: Scores
) -> Decimal:
    return sum(
        (c.weight * scores[(option.name, c.name)] for c in criteria),
        Decimal(0),
    )


# region decision_matrix
def score_matrix(
    options: Sequence[Option],
    criteria: Sequence[Criterion],
    scores: Scores,
) -> list[Ranked]:
    """Rank options by weighted total, best first; ties keep input order."""
    missing = [
        (o.name, c.name)
        for o in options
        for c in criteria
        if (o.name, c.name) not in scores
    ]
    if missing:
        raise ValueError(f"missing scores: {missing}")
    ranked = [Ranked(o.name, _total(o, criteria, scores)) for o in options]
    return sorted(ranked, key=lambda r: -r.total)


def flip_margin(
    options: Sequence[Option],
    criteria: Sequence[Criterion],
    scores: Scores,
) -> dict[str, Decimal | None]:
    """Per criterion, the smallest weight change that ties the lead.

    The change is signed (negative lowers the weight) and the weight may
    not go below zero. ``None`` means no change in that weight can
    dethrone the winner.
    """
    ranked = score_matrix(options, criteria, scores)
    winner = ranked[0]
    totals = {r.option: r.total for r in ranked}
    out: dict[str, Decimal | None] = {}
    for c in criteria:
        best: Decimal | None = None
        for o in options:
            if o.name == winner.option:
                continue
            slope = scores[(o.name, c.name)] - scores[(winner.option, c.name)]
            if slope == 0:
                continue
            delta = (totals[winner.option] - totals[o.name]) / slope
            if c.weight + delta < 0:
                continue
            if best is None or abs(delta) < abs(best):
                best = delta
        out[c.name] = best
    return out


# endregion decision_matrix


# region decision_order
def order_problems(
    criteria_at: datetime, scores_at: Sequence[datetime]
) -> list[str]:
    """Name every score entered at or before the criteria were fixed."""
    return [
        f"score {i + 1} at {t.isoformat()} is not after the criteria"
        for i, t in enumerate(scores_at)
        if t <= criteria_at
    ]


# endregion decision_order
