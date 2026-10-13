"""Chapter 57: an AI gateway, the one governed door for model calls.

Every call from every team passes through ``Gateway.handle``. Plain code
checks the team's policy (allowed models, monthly budget, requests per
minute), downgrades to a cheaper model when the budget runs short, and
writes one audit record per call with the prompt redacted. No model is
asked whether a request is allowed: a policy that a model can talk its
way past is not a policy.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal

from ai_engineer.claude.catalog import Catalog
from ai_engineer.claude.cost import call_cost
from ai_engineer.foundations.abstractions import Clock
from ai_engineer.guardrails.redact import redact
from ai_engineer.llm.model import Usage

WINDOW_SECONDS = 60.0


# region gateway_types
@dataclass(frozen=True, slots=True)
class TeamPolicy:
    team: str
    allowed_models: frozenset[str]
    monthly_budget: Decimal
    rpm: int
    fallback_model: str
    soft_limit: Decimal = Decimal("0.8")  # share of budget before downgrading


@dataclass(frozen=True, slots=True)
class Request:
    team: str
    feature: str
    model: str
    prompt: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class Decision:
    action: str  # "allow", "downgrade" or "deny"
    model: str | None
    reason: str
    cost: Decimal


@dataclass(frozen=True, slots=True)
class AuditRecord:
    time: float
    team: str
    feature: str
    action: str
    model: str | None
    reason: str
    prompt: str  # already redacted
    cost: Decimal


# endregion


@dataclass(frozen=True, slots=True)
class Service:
    name: str
    owner: str = ""
    slo: str = ""
    runbook: str = ""
    ci: bool = False
    cost_tag: str = ""


@dataclass(frozen=True, slots=True)
class Scorecard:
    service: str
    missing: tuple[str, ...]

    @property
    def on_golden_path(self) -> bool:
        return not self.missing


def spent_after(policy: TeamPolicy, spent: Decimal, cost: Decimal) -> bool:
    """True while this call keeps the team under its soft limit."""
    return spent + cost <= policy.monthly_budget * policy.soft_limit


@dataclass(slots=True)
class Gateway:
    policies: dict[str, TeamPolicy]
    catalog: Catalog
    audit: list[AuditRecord] = field(default_factory=list)
    _spend: dict[str, Decimal] = field(default_factory=dict)
    _recent: dict[str, deque[float]] = field(default_factory=dict)

    def spend(self, team: str) -> Decimal:
        return self._spend.get(team, Decimal(0))

    def _cost(self, model: str, request: Request) -> Decimal:
        usage = Usage(request.input_tokens, request.output_tokens)
        return call_cost(self.catalog.get(model), usage)

    # region gateway_handle
    def handle(self, request: Request, clock: Clock) -> Decision:
        now = clock.now()
        decision = self._decide(request, now)
        if decision.action != "deny":
            self._spend[request.team] = (
                self.spend(request.team) + decision.cost
            )
        self.audit.append(
            AuditRecord(
                now,
                request.team,
                request.feature,
                decision.action,
                decision.model,
                decision.reason,
                redact(request.prompt).text,
                decision.cost,
            )
        )
        return decision

    def _decide(self, request: Request, now: float) -> Decision:
        zero = Decimal(0)
        policy = self.policies.get(request.team)
        if policy is None:
            return Decision("deny", None, "no policy for team", zero)
        if request.model not in policy.allowed_models:
            return Decision("deny", None, "model not allowed", zero)
        window = self._recent.setdefault(request.team, deque())
        while window and now - window[0] >= WINDOW_SECONDS:
            window.popleft()
        if len(window) >= policy.rpm:
            return Decision("deny", None, "rate limit", zero)
        left = policy.monthly_budget - self.spend(request.team)
        cost = self._cost(request.model, request)
        if spent_after(policy, self.spend(request.team), cost):
            window.append(now)
            return Decision("allow", request.model, "within policy", cost)
        cheaper = self._cost(policy.fallback_model, request)
        if cheaper <= left:
            window.append(now)
            return Decision(
                "downgrade", policy.fallback_model, "budget low", cheaper
            )
        return Decision("deny", None, "budget exhausted", zero)

    # endregion

    # region gateway_scorecard
    def scorecard(self, service: Service) -> Scorecard:
        """The golden-path checks: what this service still lacks."""
        checks = {
            "owner": bool(service.owner),
            "slo": bool(service.slo),
            "runbook": bool(service.runbook),
            "ci": service.ci,
            "cost tag": service.cost_tag in self.policies,
        }
        return Scorecard(
            service.name, tuple(k for k, ok in checks.items() if not ok)
        )

    # endregion
