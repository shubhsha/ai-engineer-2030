"""Chapter 40: screen model-written tests before anyone trusts them.

A generated test is a proposal. Three filters decide, in this order,
whether it earns a place in the suite: it builds, it passes reliably on
the original code, and it kills a mutant that no accepted test killed.
The order matters, because a test that fails on the original code
"kills" every mutant and would flatter any score computed before it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ai_engineer.enterprise.evidence import Evidence, Verdict

Check = Callable[[str], list[str]]
Outcome = Literal[
    "accepted", "no_build", "fails_original", "flaky", "adds_nothing"
]


@dataclass(frozen=True, slots=True)
class Screening:
    name: str
    outcome: Outcome
    killed: frozenset[int] = frozenset()


# region screen
def build(source: str) -> Callable[[Check], None] | None:
    """Compile a candidate; it must define ``check_it(check)``."""
    scope: dict[str, Any] = {}
    try:
        exec(compile(source, "<candidate>", "exec"), scope)
    except Exception:  # a syntax error or a failing import: no build
        return None
    test: Callable[[Check], None] | None = scope.get("check_it")
    return test


def fails(test: Callable[[Check], None], target: Check) -> bool:
    try:
        test(target)
    except Exception:  # a failed assertion or a crash: the test objects
        return True
    return False


def screen(
    candidates: Sequence[tuple[str, str]],
    original: Check,
    mutants: Sequence[Check],
    *,
    known: frozenset[int] = frozenset(),
    reruns: int = 5,
) -> list[Screening]:
    """Apply the three filters in order; ``known`` are mutants already dead."""
    results, dead = [], known
    for name, source in candidates:
        test = build(source)
        if test is None:
            results.append(Screening(name, "no_build"))
            continue
        failures = sum(fails(test, original) for _ in range(reruns))
        if failures:
            outcome: Outcome = (
                "flaky" if failures < reruns else "fails_original"
            )
            results.append(Screening(name, outcome))
            continue
        killed = frozenset(i for i, m in enumerate(mutants) if fails(test, m))
        if killed <= dead:
            results.append(Screening(name, "adds_nothing", killed))
            continue
        dead |= killed
        results.append(Screening(name, "accepted", killed))
    return results


# endregion


def naive_score(
    candidates: Sequence[tuple[str, str]], mutants: Sequence[Check]
) -> float:
    """The score before filtering: every candidate that builds, as is."""
    tests = [t for _, s in candidates if (t := build(s)) is not None]
    dead = {
        i for i, m in enumerate(mutants) if any(fails(t, m) for t in tests)
    }
    return len(dead) / len(mutants)


def text_mutants(source: str, swaps: Sequence[tuple[str, str]]) -> list[str]:
    """One mutant per (old, new) text swap; each old text must be unique."""
    out = []
    for old, new in swaps:
        if source.count(old) != 1:
            raise ValueError(f"{old!r} must occur exactly once")
        out.append(source.replace(old, new))
    return out


def screening_evidence(
    screens: Sequence[Screening],
    *,
    run_id: str,
    recorded_at: float,
    scope: str,
) -> list[Evidence]:
    """Accepted tests pass; rejected ones are skipped; no build is an error."""
    verdicts: dict[Outcome, Verdict] = {
        "accepted": "pass",
        "no_build": "error",
        "fails_original": "skip",
        "flaky": "skip",
        "adds_nothing": "skip",
    }
    return [
        Evidence(
            test_id=f"generated.{s.name}",
            run_id=run_id,
            recorded_at=recorded_at,
            scope=scope,
            expected="builds, passes reliably, kills a new mutant",
            observed=f"{s.outcome}; kills {sorted(s.killed)}",
            verdict=verdicts[s.outcome],
        )
        for s in screens
    ]
