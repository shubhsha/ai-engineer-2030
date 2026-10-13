"""Chapter 45: data quality rules that run on every load.

A rule is data with an owner. Deterministic code runs it and writes one
``Evidence`` record per rule per run. A model may propose rules from a
profile, but a rule only runs after a person approves it, and the volume
check uses the median, so one old outlier cannot hide the next one.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ai_engineer.enterprise.evidence import Evidence, Verdict
from ai_engineer.foundations.abstractions import Clock
from ai_engineer.llm.model import Model, Request, user
from ai_engineer.llm.structured import generate_validated

Dimension = Literal[
    "completeness",
    "validity",
    "uniqueness",
    "consistency",
    "timeliness",
    "accuracy",
]
Severity = Literal["warn", "block", "quarantine"]
DIMENSIONS: tuple[str, ...] = (
    "completeness",
    "validity",
    "uniqueness",
    "consistency",
    "timeliness",
    "accuracy",
)
SEVERITIES: tuple[str, ...] = ("warn", "block", "quarantine")
KINDS: tuple[str, ...] = (
    "not_null",
    "range",
    "in_set",
    "pattern",
    "unique",
    "max_age",
    "not_before",
)
Row = Mapping[str, Any]


# region rule
@dataclass(frozen=True, slots=True)
class Rule:
    """One check, with an owner who answers when it fails.

    ``kind`` picks the check and ``arg`` is its parameter: ``"0:5000"``
    for ``range``, ``"a|b"`` for ``in_set``, a regular expression for
    ``pattern``, seconds for ``max_age`` and another column name for
    ``not_before`` (this column must not be earlier than that one).
    """

    name: str
    dimension: Dimension
    severity: Severity
    owner: str
    column: str
    kind: str
    arg: str = ""

    def __post_init__(self) -> None:
        if not (self.name and self.owner and self.column):
            raise ValueError("name, owner and column are required")
        if self.dimension not in DIMENSIONS:
            raise ValueError(f"dimension must be one of {DIMENSIONS}")
        if self.severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}")
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")


# endregion rule


def rules_from_json(text: str) -> tuple[Rule, ...]:
    """Read the rules file that reviewers see in a pull request."""
    items = json.loads(text)
    rules = tuple(Rule(**item) for item in items)
    names = [r.name for r in rules]
    if len(set(names)) != len(names):
        raise ValueError("rule names must be unique")
    return rules


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    rows: int
    nulls: int
    distinct: int
    minimum: Any
    maximum: Any
    top_shape: str

    @property
    def null_rate(self) -> float:
        return self.nulls / self.rows if self.rows else 0.0


def shape(value: Any) -> str:
    """Letters become A, digits become 9: 'DE-1042' is 'AA-9999'."""
    return re.sub(r"[0-9]", "9", re.sub(r"[A-Za-z]", "A", str(value)))


# region profile
def profile(rows: Sequence[Row]) -> dict[str, ColumnProfile]:
    """Per-column facts a person (or a model) can read before writing rules."""
    columns = list(dict.fromkeys(k for row in rows for k in row))
    out: dict[str, ColumnProfile] = {}
    for col in columns:
        values = [r.get(col) for r in rows]
        present = [v for v in values if v is not None]
        try:
            low, high = min(present), max(present)
        except (TypeError, ValueError):
            low = high = None
        shapes = Counter(shape(v) for v in present)
        top = shapes.most_common(1)[0][0] if shapes else ""
        out[col] = ColumnProfile(
            len(values),
            len(values) - len(present),
            len(set(map(str, present))),
            low,
            high,
            top,
        )
    return out


# endregion profile


def _violations(rule: Rule, rows: Sequence[Row], now: float) -> list[int]:
    """Indexes of rows that break ``rule``. Raises KeyError or ValueError."""
    col, bad = rule.column, []
    if rule.kind == "unique":
        seen: set[Any] = set()
        for i, row in enumerate(rows):
            value = row[col]
            if value is not None and value in seen:
                bad.append(i)
            seen.add(value)
        return bad
    for i, row in enumerate(rows):
        value = row[col]
        if rule.kind == "not_null":
            ok = value is not None
        elif value is None:
            ok = True  # a missing value is the not_null rule's business
        elif rule.kind == "range":
            low, high = rule.arg.split(":")
            ok = float(low) <= value <= float(high)
        elif rule.kind == "in_set":
            ok = str(value) in rule.arg.split("|")
        elif rule.kind == "pattern":
            ok = re.fullmatch(rule.arg, str(value)) is not None
        elif rule.kind == "max_age":
            ok = now - value <= float(rule.arg)
        else:  # not_before
            other = row[rule.arg]
            ok = other is None or value >= other
        if not ok:
            bad.append(i)
    return bad


@dataclass(frozen=True, slots=True)
class Quarantined:
    row: Row
    rules: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Run:
    evidence: tuple[Evidence, ...]
    clean: tuple[Row, ...]
    quarantined: tuple[Quarantined, ...]
    blocking: tuple[str, ...]

    @property
    def may_load(self) -> bool:
        """False when any ``block`` rule failed or could not run."""
        return not self.blocking


# region run_rules
def run_rules(
    rows: Sequence[Row], rules: Iterable[Rule], run_id: str, clock: Clock
) -> Run:
    """Run every rule, write Evidence, route rows by severity.

    ``warn`` records a failure and loads anyway. ``quarantine`` sets the
    bad rows aside and loads the rest. ``block`` stops the load. A rule
    that cannot run (a missing column) is an ``error``, never a pass.
    """
    now = clock.now()
    evidence: list[Evidence] = []
    blocking: list[str] = []
    flagged: dict[int, list[str]] = {}
    for rule in rules:
        scope = f"{rule.dimension}:{rule.column}"
        where = f"severity={rule.severity}; owner={rule.owner}"
        try:
            bad = _violations(rule, rows, now)
        except (KeyError, ValueError, TypeError) as exc:
            evidence.append(
                Evidence(
                    rule.name,
                    run_id,
                    now,
                    scope,
                    "rule runs",
                    f"{type(exc).__name__}: {exc}",
                    "error",
                    where,
                )
            )
            if rule.severity == "block":
                blocking.append(rule.name)
            continue
        verdict: Verdict = "pass" if not bad else "fail"
        evidence.append(
            Evidence(
                rule.name,
                run_id,
                now,
                scope,
                "0 violations",
                f"{len(bad)} violations in {len(rows)} rows",
                verdict,
                where,
            )
        )
        if bad and rule.severity == "block":
            blocking.append(rule.name)
        if bad and rule.severity == "quarantine":
            for i in bad:
                flagged.setdefault(i, []).append(rule.name)
    clean = tuple(r for i, r in enumerate(rows) if i not in flagged)
    held = tuple(
        Quarantined(rows[i], tuple(names))
        for i, names in sorted(flagged.items())
    )
    return Run(tuple(evidence), clean, held, tuple(blocking))


# endregion run_rules


@dataclass(frozen=True, slots=True)
class Anomaly:
    score: float
    median: float
    mad: float
    flagged: bool


# region robust_anomaly
def robust_anomaly(
    history: Sequence[tuple[int, float]],
    today: float,
    weekday: int,
    *,
    threshold: float = 3.5,
) -> Anomaly:
    """Compare today with the same weekday's past, using median and MAD.

    The score is 0.6745 * (x - median) / MAD (Iglewicz and Hoaglin), and
    3.5 is their cut-off. 0.6745 makes MAD comparable to a standard
    deviation when the data are normal. With MAD of zero any difference
    is infinitely surprising, so the score is infinite.
    """
    same = [v for day, v in history if day == weekday]
    if len(same) < 4:
        raise ValueError("need at least 4 past values for this weekday")
    med = statistics.median(same)
    mad = statistics.median(abs(v - med) for v in same)
    if mad == 0:
        score = 0.0 if today == med else math.copysign(math.inf, today - med)
    else:
        score = 0.6745 * (today - med) / mad
    return Anomaly(score, med, mad, abs(score) > threshold)


# endregion robust_anomaly


def classic_zscore(history: Sequence[float], today: float) -> float:
    """The textbook score: one mean and one deviation for every day."""
    sd = statistics.pstdev(history)
    return 0.0 if sd == 0 else (today - statistics.fmean(history)) / sd


# region triage
@dataclass(frozen=True, slots=True)
class Triage:
    rule: Rule
    status: Literal["new", "covered", "wrong", "echo"]
    reason: str


def triage(
    proposals: Sequence[Rule],
    existing: Sequence[Rule],
    trusted: Sequence[Row],
    profiles: Mapping[str, ColumnProfile],
    clock: Clock,
) -> tuple[Triage, ...]:
    """Sort model-proposed rules before a person sees them.

    covered: same column and kind as a rule that exists.
    wrong: the rule cannot run, or fails on rows people have vouched for.
    echo: a range that copies the profile's min and max. It will pass
    today by construction and so also protects any bug in today's data.
    """
    have = {(r.column, r.kind) for r in existing}
    out: list[Triage] = []
    for rule in proposals:
        prof = profiles.get(rule.column)
        if (rule.column, rule.kind) in have:
            out.append(Triage(rule, "covered", "an owner already has it"))
            continue
        run = run_rules(trusted, [rule], "triage", clock)
        if run.evidence[0].verdict != "pass":
            out.append(Triage(rule, "wrong", run.evidence[0].observed))
            continue
        if prof and rule.kind == "range" and prof.minimum is not None:
            low, high = rule.arg.split(":")
            if (float(low), float(high)) == (
                float(prof.minimum),
                float(prof.maximum),
            ):
                out.append(Triage(rule, "echo", "copies the profile"))
                continue
        out.append(Triage(rule, "new", "passes trusted rows"))
    return tuple(out)


# endregion triage


_RULE_FIELDS = ["name", "dimension", "column", "kind", "arg"]
RULE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["rules"],
    "properties": {
        "rules": {
            "type": "array",
            "maxItems": 20,
            "items": {
                "type": "object",
                "required": _RULE_FIELDS,
                "properties": {
                    "name": {"type": "string", "minLength": 3},
                    "dimension": {"enum": list(DIMENSIONS)},
                    "column": {"type": "string", "minLength": 1},
                    "kind": {"enum": list(KINDS)},
                    "arg": {"type": "string"},
                },
            },
        }
    },
}


# region propose_rules
def propose_rules(
    model: Model, profiles: Mapping[str, ColumnProfile], owner: str
) -> tuple[Rule, ...]:
    """Turn the model's suggestions into rules that nobody has approved.

    The model sees the profile (counts, min, max, top shape), never the
    rows. Severity is not its choice: every proposal starts as ``warn``
    and takes the owner given here, so a person sets the stakes.
    """
    lines = [
        f"{col}: rows={p.rows} nulls={p.nulls} distinct={p.distinct} "
        f"min={p.minimum} max={p.maximum} shape={p.top_shape}"
        for col, p in profiles.items()
    ]
    prompt = (
        "Propose data quality rules as JSON {'rules': [...]}. Fields: "
        f"{', '.join(_RULE_FIELDS)}. Kinds: {', '.join(KINDS)}.\n"
        + "\n".join(lines)
    )
    request = Request(
        "You propose rules. You never judge data.", (user(prompt),)
    )
    got = generate_validated(model, request, RULE_SCHEMA)
    return tuple(
        Rule(severity="warn", owner=owner, **item)
        for item in got.value["rules"]
    )


# endregion propose_rules
