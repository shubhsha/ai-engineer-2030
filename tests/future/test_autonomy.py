"""Tests for the autonomy guards: all offline, FakeClock, ScriptedModel."""

from __future__ import annotations

from decimal import Decimal
from itertools import cycle
from pathlib import Path

import pytest

from ai_engineer.foundations.abstractions import FakeClock
from ai_engineer.future.autonomy import (
    ActionClass,
    AutonomyPolicy,
    Budget,
    BudgetExceeded,
    Checkpoint,
    Governor,
    KillSwitch,
    NeedsApproval,
    Step,
    ToolError,
    idempotency_key,
    run_governed,
)
from ai_engineer.llm.model import Request, ScriptedModel

CENT = Decimal("0.01")


def make(
    clock: FakeClock,
    *,
    dollars: Decimal = Decimal("20"),
    steps: int = 200,
    seconds: float = 8 * 3600.0,
    tokens: int = 2_000_000,
) -> Governor:
    budget = Budget(dollars, steps, seconds, tokens)
    return Governor(budget, AutonomyPolicy(), clock)


def ok() -> None:
    return None


def test_step_budget_stops_the_201st_step(clock: FakeClock) -> None:
    gov = make(clock, steps=200)
    for _ in range(200):
        gov.step(CENT, ActionClass.READ)
    with pytest.raises(BudgetExceeded) as err:
        gov.step(CENT, ActionClass.READ)
    assert err.value.limit == "step"


def test_dollar_budget_refuses_before_spending(clock: FakeClock) -> None:
    gov = make(clock, dollars=Decimal("1.00"))
    gov.step(Decimal("0.60"), ActionClass.READ)
    with pytest.raises(BudgetExceeded) as err:
        gov.step(Decimal("0.60"), ActionClass.READ)
    assert err.value.limit == "dollar"
    assert gov.spent == Decimal("0.60")


def test_time_budget_uses_the_clock(clock: FakeClock) -> None:
    gov = make(clock, seconds=60.0)
    gov.step(CENT, ActionClass.READ)
    clock.advance(61)
    with pytest.raises(BudgetExceeded) as err:
        gov.step(CENT, ActionClass.READ)
    assert err.value.limit == "time"


def test_token_budget(clock: FakeClock) -> None:
    gov = make(clock, tokens=100)
    gov.step(CENT, ActionClass.READ, tokens=80)
    with pytest.raises(BudgetExceeded):
        gov.step(CENT, ActionClass.READ, tokens=30)


def test_kill_switch_beats_everything(clock: FakeClock) -> None:
    flag = {"stop": False}
    gov = Governor(
        Budget(), AutonomyPolicy(), clock, stop=lambda: flag["stop"]
    )
    gov.step(CENT, ActionClass.READ)
    flag["stop"] = True
    with pytest.raises(KillSwitch):
        gov.step(CENT, ActionClass.READ)


def test_policy_asks_for_external_and_irreversible(clock: FakeClock) -> None:
    gov = make(clock)
    gov.step(CENT, ActionClass.WRITE, name="save")
    with pytest.raises(NeedsApproval):
        gov.step(CENT, ActionClass.IRREVERSIBLE, name="pay")
    gov.approve("pay")
    gov.step(CENT, ActionClass.IRREVERSIBLE, name="pay")
    assert gov.steps == 2


def test_checkpoint_round_trip_and_missing(tmp_path: Path) -> None:
    cp = Checkpoint("r1", 7, Decimal("1.25"), 9)
    cp.save(tmp_path / "ck")
    assert Checkpoint.load(tmp_path / "ck", "r1") == cp
    assert Checkpoint.load(tmp_path / "ck", "nope") is None


def plan(n: int, action: ActionClass = ActionClass.READ) -> list[Step]:
    return [Step(f"s{i}", action, CENT, ok) for i in range(n)]


def test_clean_run_finishes_and_charges(
    clock: FakeClock, tmp_path: Path
) -> None:
    gov = make(clock)
    report = run_governed(plan(5), gov, tmp_path, "a")
    assert (report.status, report.attempts) == ("done", 5)
    assert report.spent == Decimal("0.05")
    assert clock.now() == 150.0


def test_scripted_model_failure_is_retried_then_succeeds(
    clock: FakeClock, tmp_path: Path
) -> None:
    model = ScriptedModel(["ERR", "ERR", "fine"])

    def call() -> None:
        reply = model.complete(Request("", ()))
        if reply.text == "ERR":
            raise ToolError("503")

    report = run_governed(
        [Step("fetch", ActionClass.READ, CENT, call)],
        make(clock),
        tmp_path,
        "b",
    )
    assert report.status == "done"
    assert report.attempts == 3
    assert len(model.requests) == 3


def bad() -> None:
    raise ToolError("payment gateway 503")


def test_unbounded_retry_is_stopped_by_the_dollar_budget(
    clock: FakeClock, tmp_path: Path
) -> None:
    gov = make(clock, dollars=Decimal("1.00"), steps=1000)
    steps = [Step("pay", ActionClass.WRITE, Decimal("0.10"), bad)]
    report = run_governed(steps, gov, tmp_path, "c", retries=None)
    assert report.status == "over_budget"
    assert report.attempts == 10
    assert "Stopped at step 0" in report.summary


def test_bounded_retries_escalate_as_stuck(
    clock: FakeClock, tmp_path: Path
) -> None:
    steps = [Step("pay", ActionClass.WRITE, CENT, bad)]
    report = run_governed(steps, make(clock), tmp_path, "d", retries=2)
    assert report.status == "stuck"
    assert report.attempts == 3


def test_loop_bug_is_stopped_by_the_step_budget(
    clock: FakeClock, tmp_path: Path
) -> None:
    gov = make(clock, steps=200)
    report = run_governed(cycle(plan(3)), gov, tmp_path, "e")
    assert report.status == "over_budget"
    assert report.attempts == 200


def test_approval_pause_then_resume_skips_finished_steps(
    clock: FakeClock, tmp_path: Path
) -> None:
    done: list[str] = []

    def mark(name: str) -> Step:
        return Step(name, ActionClass.READ, CENT, lambda: done.append(name))

    pay = Step(
        "pay", ActionClass.IRREVERSIBLE, CENT, lambda: done.append("pay")
    )
    steps = [mark("a"), mark("b"), pay, mark("c")]
    first = run_governed(steps, make(clock), tmp_path, "f")
    assert (first.status, first.next_step) == ("needs_approval", 2)
    gov = make(clock)
    gov.approve("pay")
    second = run_governed(steps, gov, tmp_path, "f")
    assert second.status == "done"
    assert done == ["a", "b", "pay", "c"]
    assert second.attempts == 4


def test_kill_flag_saves_a_checkpoint(
    clock: FakeClock, tmp_path: Path
) -> None:
    flag = {"stop": False}

    def trip() -> None:
        flag["stop"] = True

    steps = [Step("a", ActionClass.READ, CENT, trip), *plan(3)]
    gov = Governor(
        Budget(), AutonomyPolicy(), clock, stop=lambda: flag["stop"]
    )
    report = run_governed(steps, gov, tmp_path, "g")
    assert report.status == "killed"
    saved = Checkpoint.load(tmp_path, "g")
    assert saved is not None
    assert saved.next_step == 1


def test_hand_trace_a_tool_that_dies_at_step_40(
    clock: FakeClock, tmp_path: Path
) -> None:
    cost = Decimal("0.15")
    state = {"n": 0}

    def flaky() -> None:
        state["n"] += 1
        if state["n"] >= 40:
            raise ToolError("down")

    steps = [Step(f"s{i}", ActionClass.WRITE, cost, flaky) for i in range(60)]
    report = run_governed(steps, make(clock), tmp_path, "h", retries=None)
    assert report.attempts == 133
    assert report.spent == Decimal("19.95")
    assert clock.now() == 3990.0
    assert report.summary.startswith(
        "Stopped at step 39: dollar budget used: 19.95 of 20."
    )


def test_crash_after_the_effect_pays_once_with_a_key(
    clock: FakeClock, tmp_path: Path
) -> None:
    ledger: dict[str, int] = {}
    crash = {"on": True}

    def pay() -> None:
        ledger.setdefault(idempotency_key("night-1", "pay"), 38)
        if crash["on"]:
            crash["on"] = False
            raise RuntimeError("process died after the payment")

    steps = [Step("pay", ActionClass.WRITE, CENT, pay)]
    with pytest.raises(RuntimeError):
        run_governed(steps, make(clock), tmp_path, "night-1")
    assert Checkpoint.load(tmp_path, "night-1") is None
    report = run_governed(steps, make(clock), tmp_path, "night-1")
    assert report.status == "done"
    assert list(ledger.values()) == [38]
