"""Chapter 52: the cost ledger, the lever waterfall and the budget alarm.

Every call is priced with ``claude.cost.call_cost`` and tagged with a
team, a feature and an environment. A lever rewrites ledger entries and
re-prices them; the waterfall applies levers in order and keeps a lever
only when its quality guard passes. Money is ``Decimal`` throughout.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal

from ai_engineer.claude.catalog import Catalog
from ai_engineer.claude.cost import call_cost
from ai_engineer.enterprise.baseline import paired_evidence
from ai_engineer.enterprise.dq import Anomaly, robust_anomaly
from ai_engineer.enterprise.evidence import Evidence
from ai_engineer.llm.model import Usage

UNTAGGED = "(untagged)"


# region ledger
@dataclass(frozen=True, slots=True)
class Call:
    """One model call as the gateway saw it. An empty tag is missing."""

    team: str
    feature: str
    env: str
    model: str
    usage: Usage
    day: int = 0
    batch: bool = False


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    team: str
    feature: str
    env: str
    model: str
    usage: Usage
    cost: Decimal
    day: int = 0
    batch: bool = False


Ledger = tuple[LedgerEntry, ...]


def ledger_from_calls(calls: Iterable[Call], catalog: Catalog) -> Ledger:
    """Price every call from the usage the API reported."""
    return tuple(
        LedgerEntry(
            c.team,
            c.feature,
            c.env,
            c.model,
            c.usage,
            call_cost(catalog.get(c.model), c.usage, batch=c.batch),
            c.day,
            c.batch,
        )
        for c in calls
    )


def total(ledger: Sequence[LedgerEntry]) -> Decimal:
    return sum((e.cost for e in ledger), Decimal(0))


def allocate(ledger: Sequence[LedgerEntry], by: str) -> dict[str, Decimal]:
    """Spend per value of one tag, largest first. Missing tags group."""
    if by not in ("team", "feature", "env", "model"):
        raise ValueError(f"cannot allocate by {by!r}")
    sums: defaultdict[str, Decimal] = defaultdict(Decimal)
    for entry in ledger:
        sums[getattr(entry, by) or UNTAGGED] += entry.cost
    return dict(sorted(sums.items(), key=lambda kv: (-kv[1], kv[0])))


def untagged_share(ledger: Sequence[LedgerEntry]) -> Decimal:
    """The share of spend missing a team, a feature or an environment."""
    whole = total(ledger)
    if whole == 0:
        return Decimal(0)
    lost = sum(
        (e.cost for e in ledger if not (e.team and e.feature and e.env)),
        Decimal(0),
    )
    return lost / whole


def reconcile(
    ledger: Sequence[LedgerEntry],
    invoice: Decimal,
    *,
    run_id: str,
    recorded_at: float,
    tolerance: Decimal = Decimal("0.01"),
) -> Evidence:
    """Pass when the ledger is within ``tolerance`` of the invoice."""
    gap = abs(total(ledger) - invoice) / invoice
    return Evidence(
        test_id="finops.reconcile",
        run_id=run_id,
        recorded_at=recorded_at,
        scope="invoice",
        expected=f"ledger within {tolerance:.1%} of invoice {invoice}",
        observed=f"ledger {total(ledger):.2f}, gap {gap:.2%}",
        verdict="pass" if gap <= tolerance else "fail",
    )


# endregion ledger


# region lever_waterfall
@dataclass(frozen=True, slots=True)
class Lever:
    """A change to the system, and the check that it kept quality."""

    name: str
    apply: Callable[[Ledger], Ledger]
    guard: Callable[[], Evidence]


@dataclass(frozen=True, slots=True)
class Step:
    name: str
    before: Decimal
    after: Decimal
    guard: Evidence
    kept: bool

    @property
    def saving(self) -> Decimal:
        return self.before - self.after

    @property
    def share(self) -> Decimal:
        return self.saving / self.before if self.before else Decimal(0)


def waterfall(ledger: Ledger, levers: Sequence[Lever]) -> tuple[Step, ...]:
    """Apply levers in order. A lever whose guard fails changes nothing."""
    steps: list[Step] = []
    current = ledger
    for lever in levers:
        before = total(current)
        proof = lever.guard()
        kept = proof.verdict == "pass"
        candidate = lever.apply(current) if kept else current
        steps.append(Step(lever.name, before, total(candidate), proof, kept))
        current = candidate
    return tuple(steps)


def paired_guard(
    a_passed: Sequence[bool], b_passed: Sequence[bool], scope: str
) -> Callable[[], Evidence]:
    """A guard from two paired eval runs (McNemar, chapter 44)."""
    return lambda: paired_evidence(
        a_passed, b_passed, run_id="lever", recorded_at=0.0, scope=scope
    )


# endregion lever_waterfall

Match = Callable[[LedgerEntry], bool]


def _reprice(
    entry: LedgerEntry,
    catalog: Catalog,
    *,
    model: str | None = None,
    usage: Usage | None = None,
    batch: bool | None = None,
) -> LedgerEntry:
    new = replace(
        entry,
        model=model or entry.model,
        usage=usage or entry.usage,
        batch=entry.batch if batch is None else batch,
    )
    cost = call_cost(catalog.get(new.model), new.usage, batch=new.batch)
    return replace(new, cost=cost)


# region levers
def drop(match: Match) -> Callable[[Ledger], Ledger]:
    """Remove waste: calls nobody needs."""
    return lambda ledger: tuple(e for e in ledger if not match(e))


def trim_input(
    match: Match, keep: Decimal, catalog: Catalog
) -> Callable[[Ledger], Ledger]:
    """Cut tokens: keep a fraction of the input tokens."""

    def apply(ledger: Ledger) -> Ledger:
        out = []
        for e in ledger:
            if match(e):
                n = int(e.usage.input_tokens * keep)
                e = _reprice(
                    e, catalog, usage=replace(e.usage, input_tokens=n)
                )
            out.append(e)
        return tuple(out)

    return apply


def cache_prefix(
    match: Match, prefix: int, hit_rate: Decimal, catalog: Catalog
) -> Callable[[Ledger], Ledger]:
    """Move a shared prefix to the cache: on average ``hit_rate`` of
    the calls read it and the rest write it (expected-value model)."""

    def apply(ledger: Ledger) -> Ledger:
        out = []
        for e in ledger:
            if match(e) and e.usage.input_tokens >= prefix:
                reads = int(prefix * hit_rate)
                use = replace(
                    e.usage,
                    input_tokens=e.usage.input_tokens - prefix,
                    cache_read_tokens=reads,
                    cache_write_tokens=prefix - reads,
                )
                e = _reprice(e, catalog, usage=use)
            out.append(e)
        return tuple(out)

    return apply


def to_batch(match: Match, catalog: Catalog) -> Callable[[Ledger], Ledger]:
    """Batch: half price for work that can wait up to a day."""
    return lambda ledger: tuple(
        _reprice(e, catalog, batch=True) if match(e) else e for e in ledger
    )


def route(
    match: Match, model: str, catalog: Catalog
) -> Callable[[Ledger], Ledger]:
    """Route: send matching calls to a cheaper model."""
    return lambda ledger: tuple(
        _reprice(e, catalog, model=model) if match(e) else e for e in ledger
    )


# endregion levers


# region budgets
@dataclass(frozen=True, slots=True)
class BudgetAlert:
    feature: str
    spent: Decimal
    budget: Decimal
    burn: float
    level: str


def budget_alerts(
    ledger: Sequence[LedgerEntry],
    budgets: Mapping[str, Decimal],
    now: float,
    *,
    burn_limit: float = 2.0,
) -> tuple[BudgetAlert, ...]:
    """Alert on a feature that is over budget, or burning too fast.

    ``now`` is the share of the month gone, in (0, 1]. The burn rate is
    the share of budget spent divided by the share of month elapsed: 1
    spends the budget exactly at month end.
    """
    if not 0 < now <= 1:
        raise ValueError("now must be in (0, 1]")
    spent = allocate(ledger, "feature")
    alerts = []
    for feature, budget in sorted(budgets.items()):
        used = spent.get(feature, Decimal(0))
        burn = float(used / budget) / now
        if used > budget:
            alerts.append(BudgetAlert(feature, used, budget, burn, "over"))
        elif burn > burn_limit:
            alerts.append(BudgetAlert(feature, used, budget, burn, "burn"))
    return tuple(alerts)


def daily_spend(
    ledger: Sequence[LedgerEntry], start_weekday: int = 0
) -> list[tuple[int, float]]:
    """(weekday, dollars) per day, for the weekly anomaly check.

    Day 0 of the ledger falls on ``start_weekday`` (Monday is 0)."""
    days: defaultdict[int, Decimal] = defaultdict(Decimal)
    for e in ledger:
        days[e.day] += e.cost
    return [
        ((d + start_weekday) % 7, float(c)) for d, c in sorted(days.items())
    ]


def spend_anomaly(
    ledger: Sequence[LedgerEntry], today: int, start_weekday: int = 0
) -> Anomaly:
    """Median and MAD of past same-weekday spend (chapter 45)."""
    past = [e for e in ledger if e.day < today]
    dollars = float(
        sum((e.cost for e in ledger if e.day == today), Decimal(0))
    )
    weekday = (today + start_weekday) % 7
    return robust_anomaly(daily_spend(past, start_weekday), dollars, weekday)


# endregion budgets


# region break_even
def self_host_break_even(fixed_monthly: Decimal, api_per_mtok: Decimal) -> int:
    """Tokens per month at which an API bill equals a fixed cost: GPUs,
    on-call people and idle headroom, divided by the blended API price."""
    return int(fixed_monthly / api_per_mtok * 1_000_000)


# endregion break_even


# region seeded_month
_PLAN = (
    # team, feature, env, model, calls, input, output
    ("support", "support-chat", "prod", "claude-sonnet-5-5", 8000, 3000, 300),
    ("support", "triage-label", "prod", "claude-sonnet-5-5", 6000, 800, 20),
    ("data", "doc-summary", "prod", "claude-opus-5-5", 3000, 6000, 600),
    ("data", "sql-agent", "prod", "claude-opus-5-5", 1500, 4000, 800),
    ("platform", "eval-nightly", "ci", "claude-sonnet-5-5", 1000, 2000, 200),
    ("growth", "legacy-digest", "prod", "claude-sonnet-5-5", 1200, 2500, 400),
)


def synthetic_month(
    rng: random.Random, untagged: float = 0.05
) -> tuple[Call, ...]:
    """A month of calls: token counts vary by about 20%, and a random
    ``untagged`` share lost its team and feature tags (an old script)."""
    calls = []
    for team, feature, env, model, count, tin, tout in _PLAN:
        for _ in range(count):
            lost = rng.random() < untagged
            usage = Usage(
                input_tokens=round(tin * rng.uniform(0.8, 1.2)),
                output_tokens=round(tout * rng.uniform(0.8, 1.2)),
            )
            calls.append(
                Call(
                    "" if lost else team,
                    "" if lost else feature,
                    env,
                    model,
                    usage,
                    day=rng.randrange(30),
                )
            )
    return tuple(calls)


# endregion seeded_month
