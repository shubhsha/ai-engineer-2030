"""Chapter 38: run a system over graded cases and report honestly."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ai_engineer.evals.metrics import wilson_interval
from ai_engineer.llm.structured import parse_json


@dataclass(frozen=True, slots=True)
class Case:
    id: str
    input: str
    expected: Any
    tags: tuple[str, ...] = ()


Grader = Callable[[Any, Case], bool]


def _norm(value: Any) -> str:
    return " ".join(str(value).lower().split())


def exact(output: Any, case: Case) -> bool:
    """Equal after case and whitespace are normalized."""
    return _norm(output) == _norm(case.expected)


def contains(output: Any, case: Case) -> bool:
    """Every expected phrase appears in the output."""
    wanted = case.expected
    phrases = [wanted] if isinstance(wanted, str) else list(wanted)
    return all(_norm(p) in _norm(output) for p in phrases)


def json_fields(*fields: str) -> Grader:
    """The output parses as JSON and the named fields match."""

    def grade(output: Any, case: Case) -> bool:
        value, errors = parse_json(str(output))
        if errors or not isinstance(value, dict):
            return False
        return all(value.get(f) == case.expected.get(f) for f in fields)

    return grade


@dataclass(frozen=True, slots=True)
class Outcome:
    case: Case
    trial: int
    passed: bool
    output: Any
    error: str = ""


@dataclass(frozen=True, slots=True)
class Report:
    outcomes: tuple[Outcome, ...]
    trials: int

    @property
    def pass_rate(self) -> float:
        return sum(o.passed for o in self.outcomes) / len(self.outcomes)

    def case_scores(self) -> dict[str, float]:
        """Each case's share of passing trials."""
        runs: dict[str, list[bool]] = defaultdict(list)
        for o in self.outcomes:
            runs[o.case.id].append(o.passed)
        return {cid: sum(r) / len(r) for cid, r in runs.items()}

    def interval(self, z: float = 1.96) -> tuple[float, float]:
        """A confidence interval with the case as the unit.

        Trials of one case are not independent, so with several trials
        the scores are averaged per case first and the interval comes
        from the spread across cases.
        """
        scores = list(self.case_scores().values())
        if self.trials == 1:
            return wilson_interval(round(sum(scores)), len(scores), z)
        mean = statistics.fmean(scores)
        spread = statistics.stdev(scores) if len(scores) > 1 else 0.0
        half = z * spread / math.sqrt(len(scores))
        return max(0.0, mean - half), min(1.0, mean + half)

    def consistency(self) -> float:
        """Share of cases that passed every trial (observed pass^k)."""
        scores = self.case_scores().values()
        return sum(s == 1.0 for s in scores) / len(scores)

    def by_tag(self) -> dict[str, tuple[int, int]]:
        """Passed and total outcomes for each tag."""
        counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for o in self.outcomes:
            for tag in o.case.tags:
                counts[tag][0] += o.passed
                counts[tag][1] += 1
        return {tag: (p, n) for tag, (p, n) in sorted(counts.items())}

    def failures(self) -> list[Outcome]:
        return [o for o in self.outcomes if not o.passed]


def run_suite(
    system: Callable[[str], Any],
    cases: Sequence[Case],
    grader: Grader,
    *,
    trials: int = 1,
) -> Report:
    """Run every case ``trials`` times. An exception fails that trial
    and is recorded; it never stops the suite."""
    if not cases or trials < 1:
        raise ValueError("need at least one case and one trial")
    if len({c.id for c in cases}) != len(cases):
        raise ValueError("case ids must be unique")
    outcomes = []
    for case in cases:
        for trial in range(trials):
            try:
                output = system(case.input)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                outcomes.append(Outcome(case, trial, False, None, error))
                continue
            outcomes.append(Outcome(case, trial, grader(output, case), output))
    return Report(tuple(outcomes), trials)
