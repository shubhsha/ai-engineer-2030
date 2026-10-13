"""Tests for chapter 57: the AI gateway and the golden-path scorecard."""

from __future__ import annotations

from decimal import Decimal

from ai_engineer.claude.catalog import load_catalog
from ai_engineer.foundations.abstractions import FakeClock
from ai_engineer.systems.gateway import (
    Gateway,
    Request,
    Service,
    TeamPolicy,
)

BIG, SMALL = "claude-sonnet-5-5", "claude-haiku-4-5-20251001"


def policy(budget: str = "1", rpm: int = 3) -> TeamPolicy:
    return TeamPolicy(
        "search",
        frozenset({BIG, SMALL}),
        Decimal(budget),
        rpm,
        SMALL,
    )


def request(model: str = BIG, prompt: str = "hello") -> Request:
    return Request("search", "ask", model, prompt, 100_000, 10_000)


def gateway(budget: str = "1", rpm: int = 3) -> Gateway:
    return Gateway({"search": policy(budget, rpm)}, load_catalog())


def test_allow_costs_and_tracks_spend(clock: FakeClock) -> None:
    gw = gateway()
    d = gw.handle(request(), clock)
    # 100k in at $2/M plus 10k out at $10/M
    assert (d.action, d.cost) == ("allow", Decimal("0.3"))
    assert gw.spend("search") == Decimal("0.3")


def test_unknown_team_and_model_are_denied(clock: FakeClock) -> None:
    gw = gateway()
    stranger = Request("ghost", "x", BIG, "hi", 1, 1)
    assert gw.handle(stranger, clock).reason == "no policy for team"
    d = gw.handle(request("claude-fable-5-1"), clock)
    assert (d.action, d.reason) == ("deny", "model not allowed")


def test_budget_downgrades_then_denies(clock: FakeClock) -> None:
    gw = gateway(budget="0.5")
    actions = [gw.handle(request(), clock).action for _ in range(3)]
    # 0.3 allow, then 0.2 left: haiku costs 0.15, then 0.05 left: deny
    assert actions == ["allow", "downgrade", "deny"]
    assert gw.spend("search") == Decimal("0.45")


def test_sliding_window_rate_limit(clock: FakeClock) -> None:
    gw = gateway(budget="100", rpm=2)
    reasons = []
    for _ in range(3):
        reasons.append(gw.handle(request(), clock).reason)
        clock.advance(10)
    assert reasons[2] == "rate limit"
    clock.advance(41)  # the first call is now 61 s old
    assert gw.handle(request(), clock).action == "allow"


def test_denied_calls_do_not_use_the_window(clock: FakeClock) -> None:
    gw = gateway(budget="0", rpm=1)
    gw.handle(request(), clock)
    assert gw.handle(request(), clock).reason == "budget exhausted"


def test_audit_record_redacts_the_prompt(clock: FakeClock) -> None:
    gw = gateway()
    gw.handle(request(prompt="mail ana@example.com now"), clock)
    gw.handle(request("claude-fable-5-1"), clock)
    first, second = gw.audit
    assert first.prompt == "mail [EMAIL] now"
    assert (second.action, second.model) == ("deny", None)
    assert len(gw.audit) == 2


def test_scorecard_lists_what_is_missing() -> None:
    gw = gateway()
    full = Service("ask", "ana", "99.5%", "wiki/ask", True, "search")
    bare = Service("tag", cost_tag="nobody")
    assert gw.scorecard(full).on_golden_path
    card = gw.scorecard(bare)
    assert card.missing == ("owner", "slo", "runbook", "ci", "cost tag")
    assert not card.on_golden_path


def test_spend_starts_at_zero() -> None:
    assert gateway().spend("search") == Decimal(0)
