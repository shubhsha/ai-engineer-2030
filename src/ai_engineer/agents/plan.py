"""Chapter 32: plans as data, checked before anything runs."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ai_engineer.llm.model import JSON, Model, Request, user
from ai_engineer.llm.prompt import tagged
from ai_engineer.llm.structured import generate_validated


@dataclass(frozen=True, slots=True)
class Step:
    id: str
    action: str
    needs: tuple[str, ...] = ()


class CycleError(ValueError):
    pass


PLAN_SCHEMA: JSON = {
    "type": "object",
    "required": ["steps"],
    "additionalProperties": False,
    "properties": {
        "steps": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["id", "action"],
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "action": {"type": "string"},
                    "needs": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
    },
}


def waves(steps: Sequence[Step]) -> list[list[str]]:
    """Group steps into waves that each need only earlier waves.

    Steps inside a wave are independent and can run in parallel. This
    is Kahn's topological sort, one layer at a time.
    """
    waiting = {step.id: set(step.needs) for step in steps}
    done: set[str] = set()
    result: list[list[str]] = []
    while waiting:
        ready = sorted(i for i, needs in waiting.items() if needs <= done)
        if not ready:
            raise CycleError(f"steps form a cycle: {sorted(waiting)}")
        result.append(ready)
        done.update(ready)
        for i in ready:
            del waiting[i]
    return result


def check_plan(steps: Sequence[Step], actions: Collection[str]) -> list[str]:
    """Every problem with a plan, phrased so that a model can fix it."""
    ids = [step.id for step in steps]
    errors = [
        f"duplicate step id {i!r}"
        for i in sorted(set(ids))
        if ids.count(i) > 1
    ]
    for step in steps:
        if step.action not in actions:
            errors.append(f"step {step.id!r}: unknown action {step.action!r}")
        errors += [
            f"step {step.id!r} needs {need!r}, which is not a step"
            for need in step.needs
            if need not in ids
        ]
    if not errors:
        try:
            waves(steps)
        except CycleError as exc:
            errors.append(str(exc))
    return errors


def parse_plan(value: Any) -> list[Step]:
    return [
        Step(s["id"], s["action"], tuple(s.get("needs", ())))
        for s in value["steps"]
    ]


PLANNER = (
    "You plan work for an automated system. Use only the listed actions. "
    'Reply with JSON: {"steps": [{"id": ..., "action": ..., '
    '"needs": [ids of steps that must finish first]}]}.'
)


def make_plan(
    model: Model, goal: str, actions: Mapping[str, str], *, attempts: int = 3
) -> list[Step]:
    """Ask for a plan; check its shape and its logic; repair what fails."""
    catalog = "\n".join(f"- {name}: {what}" for name, what in actions.items())
    prompt = tagged("goal", goal) + "\n" + tagged("actions", catalog)
    result = generate_validated(
        model,
        Request(PLANNER, (user(prompt),)),
        PLAN_SCHEMA,
        check=lambda value: check_plan(parse_plan(value), actions),
        max_attempts=attempts,
    )
    return parse_plan(result.value)


def critical_path(steps: Sequence[Step]) -> int:
    """Length of the longest chain of steps: the fewest waves possible."""
    return len(waves(steps))


@dataclass(frozen=True, slots=True)
class RunResult:
    done: tuple[str, ...]
    failed: str | None
    replans: int


# region run_plan
def run_plan(
    steps: Sequence[Step],
    run: Callable[[Step], bool],
    replan: Callable[[frozenset[str], str], list[Step]],
    *,
    max_replans: int = 2,
) -> RunResult:
    """Run waves; when a step fails, ask for a new plan, within a budget."""
    done: list[str] = []
    replans = 0
    todo = list(steps)
    while todo:
        finished = frozenset(done)
        ready = [
            Step(
                s.id, s.action, tuple(n for n in s.needs if n not in finished)
            )
            for s in todo
        ]
        failed = None
        for wave in waves(ready)[:1]:
            for step_id in wave:
                step = next(s for s in todo if s.id == step_id)
                if run(step):
                    done.append(step_id)
                elif failed is None:
                    failed = step_id
        todo = [s for s in todo if s.id not in done]
        if failed is not None:
            if replans == max_replans:
                return RunResult(tuple(done), failed, replans)
            replans += 1
            todo = replan(frozenset(done), failed)
    return RunResult(tuple(done), None, replans)


# endregion
