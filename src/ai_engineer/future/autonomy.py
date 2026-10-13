"""Autonomy guards: budgets, a kill switch, approvals and checkpoints.

An autonomous run is a loop nobody is watching. The ``Governor`` sits
between the loop and the world: before each step it checks the stop flag,
the four budgets and the autonomy policy, and it raises one of three
exceptions when the step must not go on. ``run_governed`` turns those
exceptions into a ``RunReport`` for a person, and saves a ``Checkpoint``
after every good step so a later run can resume.
"""

from __future__ import annotations

import enum
import hashlib
import json
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from ai_engineer.foundations.abstractions import FakeClock


# region errors
class ToolError(RuntimeError):
    """A tool call failed; the step may be tried again."""


class GovernorStop(RuntimeError):
    """Base class: the run may not take this step."""


class BudgetExceeded(GovernorStop):
    def __init__(self, limit: str, used: object, cap: object) -> None:
        super().__init__(f"{limit} budget used: {used} of {cap}")
        self.limit = limit


class KillSwitch(GovernorStop):
    """A person (or a watchdog) asked the run to stop."""


class NeedsApproval(GovernorStop):
    def __init__(self, name: str, action: ActionClass) -> None:
        super().__init__(f"{action.name} step {name!r} needs approval")
        self.step_name = name
        self.action = action


# endregion errors


class ActionClass(enum.IntEnum):
    """What a step can do to the world, from safest to least reversible."""

    READ = 0
    WRITE = 1  # reversible changes inside our own systems
    EXTERNAL = 2  # visible outside: an email, a message, a ticket
    IRREVERSIBLE = 3  # money moved, data deleted, a deploy to production


@dataclass(frozen=True, slots=True)
class Budget:
    """Four independent caps; a run stops at whichever it meets first."""

    dollars: Decimal = Decimal("20")
    steps: int = 200
    seconds: float = 8 * 3600.0
    tokens: int = 2_000_000


@dataclass(frozen=True, slots=True)
class AutonomyPolicy:
    """Actions at or above ``approval_from`` need a named approval."""

    approval_from: int = ActionClass.EXTERNAL

    def needs_approval(self, action: ActionClass) -> bool:
        return action >= self.approval_from


# region governor
@dataclass(slots=True)
class Governor:
    budget: Budget
    policy: AutonomyPolicy
    clock: FakeClock
    stop: Callable[[], bool] = lambda: False
    spent: Decimal = Decimal(0)
    steps: int = 0
    tokens: int = 0
    approved: set[str] = field(default_factory=set)
    started: float = field(init=False)

    def __post_init__(self) -> None:
        self.started = self.clock.now()

    def approve(self, name: str) -> None:
        self.approved.add(name)

    def step(
        self,
        cost: Decimal,
        action: ActionClass,
        *,
        name: str = "",
        tokens: int = 0,
    ) -> None:
        """Charge one step, or raise before the step may run."""
        if self.stop():
            raise KillSwitch("stop flag is set")
        if self.policy.needs_approval(action) and name not in self.approved:
            raise NeedsApproval(name, action)
        b = self.budget
        if self.steps + 1 > b.steps:
            raise BudgetExceeded("step", self.steps, b.steps)
        waited = self.clock.now() - self.started
        if waited > b.seconds:
            raise BudgetExceeded("time", round(waited), b.seconds)
        if self.spent + cost > b.dollars:
            raise BudgetExceeded("dollar", self.spent, b.dollars)
        if self.tokens + tokens > b.tokens:
            raise BudgetExceeded("token", self.tokens, b.tokens)
        self.steps += 1
        self.spent += cost
        self.tokens += tokens


# endregion governor


# region checkpoint
@dataclass(frozen=True, slots=True)
class Checkpoint:
    """The last good step: enough to resume without redoing work."""

    run_id: str
    next_step: int
    spent: Decimal
    attempts: int

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.run_id}.json"
        tmp = path.with_suffix(".tmp")
        body = {
            "run_id": self.run_id,
            "next_step": self.next_step,
            "spent": str(self.spent),
            "attempts": self.attempts,
        }
        tmp.write_text(json.dumps(body))
        os.replace(tmp, path)  # a crash leaves the old file, never half
        return path

    @classmethod
    def load(cls, directory: Path, run_id: str) -> Checkpoint | None:
        path = directory / f"{run_id}.json"
        if not path.exists():
            return None
        raw = json.loads(path.read_text())
        return cls(
            raw["run_id"],
            int(raw["next_step"]),
            Decimal(raw["spent"]),
            int(raw["attempts"]),
        )


# endregion checkpoint


# region idempotency
def idempotency_key(run_id: str, step: str) -> str:
    """The same key on every retry and every resume of one step."""
    return hashlib.sha256(f"{run_id}:{step}".encode()).hexdigest()[:16]


# endregion idempotency


@dataclass(frozen=True, slots=True)
class Step:
    name: str
    action: ActionClass
    cost: Decimal
    run: Callable[[], None]
    seconds: float = 30.0


@dataclass(frozen=True, slots=True)
class RunReport:
    status: str  # done | killed | over_budget | needs_approval
    next_step: int
    spent: Decimal
    attempts: int
    summary: str


# region run
def run_governed(
    steps: Iterable[Step],
    governor: Governor,
    directory: Path,
    run_id: str,
    *,
    retries: int | None = 3,
) -> RunReport:
    """Run steps under ``governor``; never raise, always report.

    ``retries=None`` means retry a failing step forever, which is the
    bug the governor exists to catch.
    """
    saved = Checkpoint.load(directory, run_id)
    index = saved.next_step if saved else 0
    if saved:
        governor.spent, governor.steps = saved.spent, saved.attempts
    for i, step in enumerate(steps):
        if i < index:
            continue
        failures = 0
        while True:
            try:
                governor.step(step.cost, step.action, name=step.name)
            except GovernorStop as stop:
                return _escalate(stop, governor, directory, run_id, i)
            governor.clock.advance(step.seconds)
            try:
                step.run()
                break
            except ToolError:
                failures += 1
                if retries is not None and failures > retries:
                    msg = f"step {step.name!r} failed {failures} times"
                    return _escalate(
                        GovernorStop(msg), governor, directory, run_id, i
                    )
        Checkpoint(run_id, i + 1, governor.spent, governor.steps).save(
            directory
        )
    return RunReport("done", index, governor.spent, governor.steps, "all done")


def _escalate(
    stop: GovernorStop,
    governor: Governor,
    directory: Path,
    run_id: str,
    at: int,
) -> RunReport:
    kinds = {
        KillSwitch: "killed",
        NeedsApproval: "needs_approval",
        BudgetExceeded: "over_budget",
    }
    status = kinds.get(type(stop), "stuck")
    Checkpoint(run_id, at, governor.spent, governor.steps).save(directory)
    summary = (
        f"Stopped at step {at}: {stop}. Spent {governor.spent} dollars "
        f"in {governor.steps} attempts. Resume continues at step {at}."
    )
    return RunReport(status, at, governor.spent, governor.steps, summary)


# endregion run
