"""Chapter 35: a lead agent that briefs workers in clean contexts.

The lead splits a question into briefs, each worker researches one
brief in a context of its own, and the lead writes the answer from the
findings. Spreading work across contexts multiplies the tokens spent,
so use it when the parts are independent and the answer is worth it.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from ai_engineer.llm.model import Model, Reply, Request, Usage, user
from ai_engineer.llm.prompt import tagged
from ai_engineer.llm.structured import generate_validated

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Brief:
    """A task for a worker that starts with no other context."""

    objective: str
    output: str = "Findings as short bullet points, each with its source."
    boundaries: str = "Stay inside the objective. Say what you could not find."

    def render(self) -> str:
        return "\n\n".join(
            [
                tagged("objective", self.objective),
                tagged("output_format", self.output),
                tagged("boundaries", self.boundaries),
            ]
        )


_FIELD = {"type": "string"}
BRIEFS_SCHEMA = {
    "type": "object",
    "properties": {
        "briefs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "objective": _FIELD,
                    "output": _FIELD,
                    "boundaries": _FIELD,
                },
                "required": ["objective", "output", "boundaries"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["briefs"],
    "additionalProperties": False,
}

PLANNER = (
    "You lead a research team. Split the question into parts that can "
    "be researched independently. For each part write a brief: the "
    "objective, the output you need back and the boundaries of the task. "
    "A worker sees only its brief, so make each one self-contained."
)
WORKER = (
    "You research one brief. Follow its boundaries and output format. "
    "Report what you found and how sure you are, not what you assume."
)
LEAD = (
    "Answer the question from the findings below. Say where findings "
    "disagree or are missing. Do not add facts they do not contain."
)


@dataclass(frozen=True, slots=True)
class Research:
    answer: str
    findings: tuple[str, ...]
    lead_usage: Usage
    worker_usage: Usage

    @property
    def usage(self) -> Usage:
        return self.lead_usage + self.worker_usage


def fan_out(
    model: Model,
    briefs: Sequence[Brief],
    *,
    system: str = WORKER,
    limit: int = 3,
) -> list[Reply]:
    """Run each brief in a fresh context, at most ``limit`` at a time.

    A worker that fails returns a note instead of raising, so that one
    failure leaves the lead with partial findings rather than none.
    """

    def work(brief: Brief) -> Reply:
        try:
            return model.complete(Request(system, (user(brief.render()),)))
        except Exception as exc:
            log.warning("worker failed on %r: %s", brief.objective, exc)
            return Reply(f"[worker failed: {type(exc).__name__}]", (), "error")

    with ThreadPoolExecutor(max_workers=limit) as pool:
        return list(pool.map(work, briefs))


def research(
    lead: Model,
    worker: Model,
    question: str,
    *,
    max_workers: int = 3,
    limit: int = 3,
) -> Research:
    """Plan briefs with ``lead``, research them with ``worker``, then
    have ``lead`` write the answer from the findings alone."""

    def check(value: Any) -> list[str]:
        n = len(value["briefs"])
        if 1 <= n <= max_workers:
            return []
        return [f"$.briefs: write 1 to {max_workers} briefs, not {n}"]

    asked = user(tagged("question", question))
    plan = generate_validated(
        lead, Request(PLANNER, (asked,)), BRIEFS_SCHEMA, check=check
    )
    briefs = [Brief(**b) for b in plan.value["briefs"]]
    replies = fan_out(worker, briefs, limit=limit)
    findings = tuple(reply.text for reply in replies)
    report = "\n\n".join(
        tagged("finding", f"objective: {brief.objective}\n{text}")
        for brief, text in zip(briefs, findings, strict=True)
    )
    prompt = f"{tagged('question', question)}\n\n{report}"
    final = lead.complete(Request(LEAD, (user(prompt),)))
    worker_usage = sum((reply.usage for reply in replies), Usage())
    return Research(
        final.text, findings, plan.usage + final.usage, worker_usage
    )


# region amdahl
def amdahl(serial: float, workers: int) -> float:
    """Best speed-up when a fraction ``serial`` of the work cannot be
    shared and the rest is split evenly across ``workers``."""
    return 1 / (serial + (1 - serial) / workers)


# endregion


# region timed_model
class TimedModel:
    """Wrap a model with a virtual clock, so that wall time can be
    tested without sleeping. Each call takes ``seconds`` and runs on the
    first of ``slots`` parallel slots to come free."""

    def __init__(self, inner: Model, seconds: float, slots: int = 1) -> None:
        self._inner = inner
        self._seconds = seconds
        self._free = [0.0] * slots
        self._lock = threading.Lock()

    def complete(self, request: Request) -> Reply:
        with self._lock:
            slot = self._free.index(min(self._free))
            self._free[slot] += self._seconds
        return self._inner.complete(request)

    @property
    def elapsed(self) -> float:
        return max(self._free)


# endregion


# region audit
def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def audit(
    briefs: Sequence[Brief], findings: Sequence[str], *, same: float = 0.6
) -> list[str]:
    """Checks the lead can run before it trusts the findings: briefs
    that overlap, findings that leave the bullet format, and bullets
    with no ``[source]``. Returns one message per problem."""
    problems = []
    for i, a in enumerate(briefs):
        for j in range(i + 1, len(briefs)):
            wa, wb = _words(a.objective), _words(briefs[j].objective)
            if len(wa & wb) / len(wa | wb) >= same:
                problems.append(f"briefs {i} and {j} overlap")
    for i, text in enumerate(findings):
        for line in filter(None, (x.strip() for x in text.splitlines())):
            if not line.startswith("- "):
                problems.append(f"finding {i}: not a bullet: {line[:30]!r}")
            elif "[" not in line:
                problems.append(f"finding {i}: no source: {line[:30]!r}")
    return problems


# endregion


# region agent_card
@dataclass(frozen=True, slots=True)
class AgentCard:
    """What a remote agent publishes about itself, reduced to the
    fields a client needs in order to choose it."""

    name: str
    url: str
    skills: tuple[str, ...]
    auth: str = "bearer"


def discover(cards: Sequence[AgentCard], skill: str) -> list[AgentCard]:
    """Cards whose agents offer ``skill``, in the order given."""
    return [card for card in cards if skill in card.skills]


# endregion
