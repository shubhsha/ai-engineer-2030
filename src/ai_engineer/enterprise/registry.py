"""Chapter 50: an AI system inventory, risk tiers, controls and evidence.

Governance is bookkeeping that a reviewer can check. The inventory says
what exists, a tier says how much care it needs, a control says what must
be true, and an Evidence record says when someone last saw it be true.
Nothing here decides what is lawful; that is a person's job.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from ai_engineer.enterprise.evidence import Evidence

DAY = 86_400.0
TIERS: tuple[str, ...] = ("prohibited", "high", "limited", "minimal")


# region system
@dataclass(frozen=True, slots=True)
class AiSystem:
    """One row of the inventory: who owns it, what it does, who it touches."""

    system_id: str
    owner: str
    purpose: str
    use_area: str
    data_categories: tuple[str, ...]
    affected: tuple[str, ...]
    model_ids: tuple[str, ...]
    human_oversight: str = "none"

    def __post_init__(self) -> None:
        if not (self.system_id and self.owner and self.purpose):
            raise ValueError("system_id, owner and purpose are required")
        if self.human_oversight not in ("none", "review", "approve"):
            raise ValueError("oversight is none, review or approve")


# endregion system


# region tier
@dataclass(frozen=True, slots=True)
class TierRule:
    """If any listed value is present, the system is at least this tier."""

    tier: str
    reason: str
    use_areas: tuple[str, ...] = ()
    data_categories: tuple[str, ...] = ()
    affected: tuple[str, ...] = ()

    def matches(self, system: AiSystem) -> bool:
        return (
            system.use_area in self.use_areas
            or bool(set(system.data_categories) & set(self.data_categories))
            or bool(set(system.affected) & set(self.affected))
        )


@dataclass(frozen=True, slots=True)
class TierResult:
    tier: str
    reason: str


DEFAULT_RULES: tuple[TierRule, ...] = (
    TierRule("prohibited", "social scoring", use_areas=("social-scoring",)),
    TierRule(
        "high",
        "employment, credit or education decisions about people",
        use_areas=("employment", "credit", "education", "essential-services"),
    ),
    TierRule(
        "limited",
        "talks to the public or touches personal data",
        data_categories=("personal", "special"),
        affected=("customers", "public"),
    ),
)


def risk_tier(
    system: AiSystem, rules: Sequence[TierRule] = DEFAULT_RULES
) -> TierResult:
    """The strictest tier whose rule matches; minimal when none does."""
    best = TierResult("minimal", "no rule matched")
    for rule in rules:
        if rule.tier not in TIERS:
            raise ValueError(f"unknown tier {rule.tier!r}")
        if rule.matches(system) and TIERS.index(rule.tier) < TIERS.index(
            best.tier
        ):
            best = TierResult(rule.tier, rule.reason)
    return best


# endregion tier


# region control
@dataclass(frozen=True, slots=True)
class Control:
    """A statement that must stay true, and the proof that it is."""

    control_id: str
    title: str
    frameworks: tuple[str, ...]
    evidence_kinds: tuple[str, ...]
    max_age_days: float
    tiers: tuple[str, ...] = ("high",)


def kind_of(item: Evidence) -> str:
    """Evidence ids look like ``kind:detail``; the kind is the left part."""
    return item.test_id.split(":", 1)[0]


def controls_for(
    tier: str, controls: Iterable[Control]
) -> tuple[Control, ...]:
    return tuple(c for c in controls if tier in c.tiers)


# endregion control


# region coverage
@dataclass(frozen=True, slots=True)
class Coverage:
    covered: tuple[tuple[str, str], ...]
    stale: tuple[tuple[str, str], ...]
    failing: tuple[tuple[str, str], ...]
    missing: tuple[tuple[str, str], ...]

    def total(self) -> int:
        return (
            len(self.covered)
            + len(self.stale)
            + len(self.failing)
            + len(self.missing)
        )

    def share_covered(self) -> float:
        return len(self.covered) / self.total() if self.total() else 1.0


def control_status(
    system: AiSystem,
    control: Control,
    evidence: Sequence[Evidence],
    now: float,
) -> str:
    """covered, stale, failing or missing. Only fresh passes count."""
    latest: dict[str, Evidence] = {}
    for item in sorted(evidence, key=lambda e: e.recorded_at):
        if item.scope == system.system_id:
            latest[kind_of(item)] = item
    states = set()
    for kind in control.evidence_kinds:
        found = latest.get(kind)
        if found is None or found.verdict in ("skip", "error"):
            states.add("missing")
        elif found.verdict == "fail":
            states.add("failing")
        elif now - found.recorded_at > control.max_age_days * DAY:
            states.add("stale")
    for state in ("missing", "failing", "stale"):
        if state in states:
            return state
    return "covered"


def coverage(
    systems: Iterable[AiSystem],
    controls: Sequence[Control],
    evidence: Sequence[Evidence],
    now: float,
    rules: Sequence[TierRule] = DEFAULT_RULES,
) -> Coverage:
    """Check every control that applies to each system's tier."""
    buckets: dict[str, list[tuple[str, str]]] = {
        "covered": [],
        "stale": [],
        "failing": [],
        "missing": [],
    }
    for system in systems:
        tier = risk_tier(system, rules).tier
        for control in controls_for(tier, controls):
            status = control_status(system, control, evidence, now)
            buckets[status].append((system.system_id, control.control_id))
    return Coverage(
        tuple(buckets["covered"]),
        tuple(buckets["stale"]),
        tuple(buckets["failing"]),
        tuple(buckets["missing"]),
    )


# endregion coverage


# region pack
def _day(at: float) -> str:
    return datetime.fromtimestamp(at, UTC).date().isoformat()


@dataclass(frozen=True, slots=True)
class Event:
    at: float
    text: str


PACK_SECTIONS: tuple[str, ...] = (
    "system_description",
    "risk_assessment",
    "evaluation_results",
    "human_oversight",
    "incident_log",
    "change_history",
)


def audit_pack(
    system: AiSystem,
    controls: Sequence[Control],
    evidence: Sequence[Evidence],
    now: float,
    *,
    incidents: Sequence[Event] = (),
    changes: Sequence[Event] = (),
    rules: Sequence[TierRule] = DEFAULT_RULES,
) -> dict[str, str]:
    """The six documents a reviewer asks for, drawn from the records."""
    result = risk_tier(system, rules)
    rows = []
    for control in controls_for(result.tier, controls):
        status = control_status(system, control, evidence, now)
        rows.append(
            f"{control.control_id} ({', '.join(control.frameworks)}): {status}"
        )
    proofs = sorted(
        (e for e in evidence if e.scope == system.system_id),
        key=lambda e: e.recorded_at,
    )
    return {
        "system_description": (
            f"{system.system_id}: {system.purpose}. Owner: {system.owner}. "
            f"Models: {', '.join(system.model_ids)}. "
            f"Data: {', '.join(system.data_categories) or 'none'}."
        ),
        "risk_assessment": (
            f"Tier {result.tier} because {result.reason}. "
            f"Affected: {', '.join(system.affected) or 'nobody outside'}.\n"
            + "\n".join(rows)
        ),
        "evaluation_results": "\n".join(
            f"{_day(e.recorded_at)} {e.test_id} {e.verdict} "
            f"expected={e.expected} observed={e.observed} "
            f"proof={e.location}"
            for e in proofs
        ),
        "human_oversight": f"Oversight level: {system.human_oversight}.",
        "incident_log": "\n".join(f"{_day(e.at)} {e.text}" for e in incidents),
        "change_history": "\n".join(f"{_day(e.at)} {e.text}" for e in changes),
    }


def render_pack(pack: dict[str, str]) -> str:
    return "\n\n".join(f"## {name}\n{pack[name]}" for name in PACK_SECTIONS)


# endregion pack
