"""Learning tools: SM-2 flash cards, a review queue, a year plan, and a
toy forgetting-curve learner for comparing massed and spaced practice."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date, timedelta

MIN_EASE = 1.3


# region card_review
@dataclass(frozen=True, slots=True)
class Card:
    """One fact to remember. Dates come from the caller, never the clock."""

    front: str
    back: str
    due: date
    interval: int = 0
    ease: float = 2.5
    reps: int = 0


def review(card: Card, quality: int, today: date) -> Card:
    """SM-2 (after Wozniak, 1990): quality 0-5, 3 or more is a pass."""
    if not 0 <= quality <= 5:
        raise ValueError("quality must be between 0 and 5")
    miss = 5 - quality
    ease = max(MIN_EASE, card.ease + 0.1 - miss * (0.08 + miss * 0.02))
    if quality < 3:
        return replace(card, reps=0, interval=1, due=today + timedelta(1))
    reps = card.reps + 1
    if reps == 1:
        interval = 1
    elif reps == 2:
        interval = 6
    else:
        interval = round(card.interval * card.ease)
    due = today + timedelta(days=interval)
    return replace(card, reps=reps, interval=interval, ease=ease, due=due)


def due_today(cards: list[Card], today: date) -> list[Card]:
    """The review queue: overdue cards first, then by front."""
    due = [c for c in cards if c.due <= today]
    return sorted(due, key=lambda c: (c.due, c.front))


# endregion card_review


# region year_plan
@dataclass(frozen=True, slots=True)
class Quarter:
    start: date
    skill: str
    project: str
    practice: str


PRACTICES = {
    "writing": ("one design doc, reviewed by two people", "write 200 words"),
    "speaking": ("give a talk to your team", "explain one idea aloud"),
    "domain": ("shadow a user for a day", "read one domain case a week"),
    "system-design": (
        "redesign one system you run, on paper",
        "one design critique a week",
    ),
    "economics": ("price one feature end to end", "read one cost report"),
}
DEFAULT = ("ship one project that needs this skill", "30 minutes daily")


def plan_year(gaps: Mapping[str, int], start: date) -> tuple[Quarter, ...]:
    """Four quarters, one skill each: the four largest gaps, ties by name.
    Every quarter has a project and a daily practice; more than four
    goals are rejected, so the plan cannot outgrow the year."""
    if not gaps:
        raise ValueError("no gaps to plan")
    ranked = sorted(
        (g for g in gaps.items() if g[1] > 0), key=lambda g: (-g[1], g[0])
    )
    quarters = []
    for i, (skill, _) in enumerate(ranked[:4]):
        project, practice = PRACTICES.get(skill, DEFAULT)
        begin = start + timedelta(days=91 * i)
        quarters.append(Quarter(begin, skill, project, practice))
    return tuple(quarters)


# endregion year_plan


# region forgetting
def retrievability(elapsed: float, stability: float) -> float:
    """Ebbinghaus-style forgetting curve: exp(-t / s), s in days."""
    return math.exp(-elapsed / stability)


def _study(stab: float, elapsed: float, rng: random.Random) -> float:
    """Return the new stability after one review attempt."""
    r = retrievability(elapsed, stab)
    if rng.random() < r:
        return stab * (1.2 + 2.0 * (1.0 - r))
    return max(1.0, stab * 0.5)


# endregion forgetting


# region learner_sim


def simulate(
    rng: random.Random,
    spaced: bool,
    n_cards: int = 200,
    horizon: int = 30,
    reviews: int = 4,
) -> tuple[float, float]:
    """Mean recall at the horizon and mean reviews per card. Spaced
    reviews follow SM-2 (pass if the card was recalled); massed reviews
    all happen on day 0, the same number per card."""
    start = date(2030, 1, 1)
    stab = [1.0] * n_cards
    last = [0.0] * n_cards
    if not spaced:
        for i in range(n_cards):
            for _ in range(reviews):
                stab[i] = _study(stab[i], 0.05, rng)
        return _recall(stab, last, horizon), float(reviews)
    cards = [Card(f"c{i}", "", start) for i in range(n_cards)]
    done = 0
    for day in range(horizon):
        today = start + timedelta(days=day)
        for card in due_today(cards, today):
            i = int(card.front[1:])
            before = stab[i]
            stab[i] = _study(before, day - last[i], rng)
            last[i] = float(day)
            quality = 4 if stab[i] > before else 2
            cards[i] = review(card, quality, today)
            done += 1
    return _recall(stab, last, horizon), done / n_cards


def _recall(stab: list[float], last: list[float], horizon: int) -> float:
    total = sum(
        retrievability(horizon - t, s) for s, t in zip(stab, last, strict=True)
    )
    return total / len(stab)


# endregion learner_sim
