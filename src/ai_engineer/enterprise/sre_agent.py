"""Chapter 48: an AI SRE that reads telemetry and proposes, never acts.

The agent has read-only tools over metrics, traces, deploy records and
runbooks. Its answer is a list of hypotheses and a list of proposed
actions. Code, not the model, decides which proposals are allowed, and
every allowed proposal still waits for a named human.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ai_engineer.agents.loop import run_agent
from ai_engineer.llm.model import Model
from ai_engineer.llm.structured import parse_json
from ai_engineer.llm.tools import Tool, ToolError, ToolRegistry

Value = str | int | float


# region telemetry
@dataclass(frozen=True, slots=True)
class Span:
    """One timed step. Attribute names follow the gen_ai.* conventions,
    for example ``gen_ai.request.model`` and ``gen_ai.usage.input_tokens``."""

    trace_id: str
    span_id: str
    parent_id: str
    name: str
    start: float
    end: float
    status: str = "ok"  # "ok" or "error"
    attributes: tuple[tuple[str, Value], ...] = ()

    def attr(self, key: str, default: Value = "") -> Value:
        return dict(self.attributes).get(key, default)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class Deploy:
    at: float
    service: str
    version: str
    change: str


@dataclass(frozen=True, slots=True)
class Telemetry:
    metrics: dict[str, tuple[tuple[float, float], ...]]
    traces: dict[str, tuple[Span, ...]]
    deploys: tuple[Deploy, ...]
    runbooks: dict[str, str]


@dataclass(frozen=True, slots=True)
class Alert:
    name: str
    fired_at: float
    metric: str
    summary: str


# endregion telemetry


# region sre_tools
def build_tools(data: Telemetry) -> ToolRegistry:
    """Four tools, all read-only. None of them can change anything."""

    def query_metric(metric: str, start: float, end: float) -> Any:
        if metric not in data.metrics:
            have = sorted(data.metrics)
            raise ToolError(f"No metric {metric!r}. Have: {have}")
        return [[t, v] for t, v in data.metrics[metric] if start <= t <= end]

    def fetch_trace(trace_id: str) -> Any:
        if trace_id not in data.traces:
            raise ToolError(f"No trace {trace_id!r}.")
        return [
            {
                "span": s.span_id,
                "name": s.name,
                "seconds": round(s.duration, 3),
                "status": s.status,
                "attributes": dict(s.attributes),
            }
            for s in data.traces[trace_id]
        ]

    def recent_deploys(before: float, hours: float) -> Any:
        return [
            {"at": d.at, "service": d.service, "version": d.version}
            for d in data.deploys
            if before - hours * 3600 <= d.at <= before
        ]

    def read_runbook(runbook: str) -> str:
        if runbook not in data.runbooks:
            have = sorted(data.runbooks)
            raise ToolError(f"No runbook {runbook!r}. Have: {have}")
        return data.runbooks[runbook]

    def tool(fn: Any, doc: str, props: dict[str, str]) -> Tool:
        schema = {
            "type": "object",
            "properties": {k: {"type": v} for k, v in props.items()},
            "required": list(props),
        }
        return Tool(fn.__name__, doc, schema, fn, read_only=True)

    return ToolRegistry(
        [
            tool(
                query_metric,
                "Points (time, value) of one metric.",
                {"metric": "string", "start": "number", "end": "number"},
            ),
            tool(
                fetch_trace, "The spans of one trace.", {"trace_id": "string"}
            ),
            tool(
                recent_deploys,
                "Deploys in the hours before a time.",
                {"before": "number", "hours": "number"},
            ),
            tool(
                read_runbook, "The text of one runbook.", {"runbook": "string"}
            ),
        ]
    )


# endregion sre_tools


# region sre_propose
ALLOWED_ACTIONS = frozenset(
    {"rollback", "restart", "scale_up", "run_check", "page_owner"}
)

SYSTEM = (
    "You are an SRE assistant. You may only read: use the tools. You "
    "cannot change anything. Finish with JSON only: "
    '{"hypotheses": [{"cause": str, "evidence": str}], '
    '"actions": [{"action": str, "target": str, "reason": str}]}.'
)


@dataclass(frozen=True, slots=True)
class Hypothesis:
    cause: str
    evidence: str


@dataclass(frozen=True, slots=True)
class Proposal:
    action: str
    target: str
    reason: str
    status: str  # "needs_approval" or "blocked"


@dataclass(frozen=True, slots=True)
class Report:
    hypotheses: tuple[Hypothesis, ...]
    proposals: tuple[Proposal, ...]
    stop: str  # "end_turn", "unparseable", "max_steps", ...
    steps: int


def check_actions(raw: Sequence[Any]) -> tuple[Proposal, ...]:
    """Allowed actions wait for a human; anything else is blocked."""
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", ""))
        status = "needs_approval" if action in ALLOWED_ACTIONS else "blocked"
        out.append(
            Proposal(
                action,
                str(item.get("target", "")),
                str(item.get("reason", "")),
                status,
            )
        )
    return tuple(out)


def hypothesize(
    model: Model, alert: Alert, data: Telemetry, *, max_steps: int = 6
) -> Report:
    task = (
        f"Alert {alert.name} fired at {alert.fired_at}: {alert.summary} "
        f"Metric: {alert.metric}. Find the likely cause."
    )
    result = run_agent(
        model, task, build_tools(data), system=SYSTEM, max_steps=max_steps
    )
    parsed, errors = parse_json(result.answer)
    if errors or not isinstance(parsed, dict):
        return Report((), (), "unparseable", result.steps)
    hypotheses = tuple(
        Hypothesis(str(h.get("cause", "")), str(h.get("evidence", "")))
        for h in parsed.get("hypotheses", [])
        if isinstance(h, dict)
    )
    proposals = check_actions(parsed.get("actions", []))
    return Report(hypotheses, proposals, result.stop, result.steps)


@dataclass(frozen=True, slots=True)
class Approval:
    proposal: Proposal
    approved_by: str


def approve(proposal: Proposal, by: str) -> Approval:
    """The only way to turn a proposal into something a runner may do."""
    if proposal.status != "needs_approval":
        raise PermissionError(f"{proposal.action!r} is not an allowed action")
    if not by.strip():
        raise PermissionError("an approval needs a named human")
    return Approval(proposal, by)


# endregion sre_propose


# region postmortem
@dataclass(frozen=True, slots=True)
class TimelineEntry:
    at: float
    kind: str  # "deploy", "error" or "slow"
    text: str


def postmortem_timeline(
    trace: Sequence[Span],
    deploys: Sequence[Deploy],
    *,
    lookback_s: float = 3600.0,
) -> tuple[TimelineEntry, ...]:
    """Facts for a postmortem draft, in time order, from records only."""
    if not trace:
        return ()
    first = min(s.start for s in trace)
    last = max(s.end for s in trace)
    rows = [
        TimelineEntry(d.at, "deploy", f"{d.service} {d.version}: {d.change}")
        for d in deploys
        if first - lookback_s <= d.at <= last
    ]
    rows += [
        TimelineEntry(s.start, "error", f"{s.name} failed ({s.span_id})")
        for s in trace
        if s.status == "error"
    ]
    slowest = max(trace, key=lambda s: s.duration)
    rows.append(
        TimelineEntry(
            slowest.start,
            "slow",
            f"{slowest.name} took {slowest.duration:.1f}s",
        )
    )
    return tuple(sorted(rows, key=lambda r: (r.at, r.kind)))


# endregion postmortem
