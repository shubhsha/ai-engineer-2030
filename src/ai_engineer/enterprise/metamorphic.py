"""Chapter 40: metamorphic relations that write evidence.

When no one can say what the right answer is, you can often still say how
two answers must relate. A relation changes the input in a way whose
effect on the output is known, runs the system twice and checks the pair.
Each relation writes one ``Evidence`` record, so a release gate can read
it next to every other test.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Generic, TypeVar

from ai_engineer.enterprise.evidence import Evidence, Verdict

X = TypeVar("X")
Y = TypeVar("Y")


# region relation
@dataclass(frozen=True, slots=True)
class Relation(Generic[X, Y]):
    """How a change to the input must show up in the output."""

    name: str
    transform: Callable[[X], X]
    check: Callable[[X, Y, Y], bool]  # (input, output, follow-up output)


def run_relations(
    system: Callable[[X], Y],
    relations: Sequence[Relation[X, Y]],
    inputs: Sequence[X],
    *,
    run_id: str,
    recorded_at: float,
    scope: str,
) -> list[Evidence]:
    """One record per relation: fail if any input breaks it."""
    records = []
    for relation in relations:
        broken: list[str] = []
        verdict: Verdict = "pass"
        observed = f"held on {len(inputs)} inputs"
        try:
            for item in inputs:
                source = system(item)
                follow = system(relation.transform(item))
                if not relation.check(item, source, follow):
                    broken.append(repr(item)[:60])
        except Exception as exc:  # the check could not run: not a pass
            verdict, observed = "error", f"{type(exc).__name__}: {exc}"
        else:
            if broken:
                verdict = "fail"
                observed = f"broken on {len(broken)}: {broken[0]}"
        records.append(
            Evidence(
                test_id=f"metamorphic.{relation.name}",
                run_id=run_id,
                recorded_at=recorded_at,
                scope=scope,
                expected=f"{relation.name} holds on {len(inputs)} inputs",
                observed=observed,
                verdict=verdict,
            )
        )
    return records


# endregion


# region subjects
@dataclass(frozen=True, slots=True)
class Sale:
    month: int
    day: int
    amount: Decimal


def revenue_by_month(sales: Iterable[Sale]) -> dict[int, Decimal]:
    totals: dict[int, Decimal] = {}
    for sale in sales:
        totals[sale.month] = totals.get(sale.month, Decimal(0)) + sale.amount
    return totals


def revenue_by_month_buggy(sales: Iterable[Sale]) -> dict[int, Decimal]:
    """Fault: a sale on March 1 also lands in February (inclusive edge)."""
    rows = tuple(sales)
    totals = revenue_by_month(rows)
    for sale in rows:
        if (sale.month, sale.day) == (3, 1):
            totals[2] = totals.get(2, Decimal(0)) + sale.amount
    return totals


def classify_ticket(text: str) -> str:
    words = text.lower().split()
    if "invoice" in words or "refund" in words:
        return "billing"
    return "bug" if "error" in words or "crash" in words else "other"


def classify_ticket_buggy(text: str) -> str:
    """Fault: a trailing space flips the label to ``other``."""
    return "other" if text.endswith(" ") else classify_ticket(text)


# endregion


# region relations
MONTHS_ADD_UP: Relation[tuple[Sale, ...], dict[int, Decimal]] = Relation(
    "months_add_up",
    lambda sales: sales,
    lambda sales, out, _: (
        sum(out.values(), Decimal(0))
        == sum((s.amount for s in sales), Decimal(0))
    ),
)
CASE_AND_SPACE: Relation[str, str] = Relation(
    "case_and_space_invariance",
    lambda text: text.upper() + "  ",
    lambda _text, label, follow: label == follow,
)
# endregion


def random_sales(
    rng: random.Random, count: int, *, edges: bool = False
) -> tuple[Sale, ...]:
    """Synthetic sales; ``edges`` puts a third of them on the first day."""
    return tuple(
        Sale(
            rng.randint(1, 12),
            1 if edges and rng.random() < 1 / 3 else rng.randint(1, 28),
            Decimal(rng.randint(1, 999)),
        )
        for _ in range(count)
    )
