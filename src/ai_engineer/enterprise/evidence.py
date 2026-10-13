"""Chapter 40: the evidence record that every Volume IV test writes.

A check that leaves no record did not happen, as far as a release review
is concerned. A row count, a reconciliation and a model-graded answer
all write one ``Evidence`` record of the same shape, so one gate and one
dashboard can read every kind of test.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

Verdict = Literal["pass", "fail", "error", "skip"]
VERDICTS: tuple[Verdict, ...] = ("pass", "fail", "error", "skip")


@dataclass(frozen=True, slots=True)
class Evidence:
    """What one test run expected, what it saw and what it concluded.

    ``error`` means the check itself could not run, which is not a pass.
    ``location`` says where the full proof lives: a file, a query or a
    trace id that a reviewer can open.
    """

    test_id: str
    run_id: str
    recorded_at: float
    scope: str
    expected: str
    observed: str
    verdict: Verdict
    location: str = ""

    def __post_init__(self) -> None:
        if not (self.test_id and self.run_id and self.scope):
            raise ValueError("test_id, run_id and scope are required")
        if self.verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}")


@dataclass(frozen=True, slots=True)
class Gate:
    passed: bool
    counts: tuple[tuple[Verdict, int], ...]
    blocking: tuple[str, ...]


def release_gate(
    evidence: Iterable[Evidence], *, required: Iterable[str]
) -> Gate:
    """Open only when every required test passed and nothing failed.

    The latest record per test wins, so a rerun replaces a flaky result
    instead of hiding it. A required test with no record blocks: missing
    evidence is not good evidence.
    """
    latest: dict[str, Evidence] = {}
    for item in sorted(evidence, key=lambda e: e.recorded_at):
        latest[item.test_id] = item
    blocking = {
        test_id
        for test_id, item in latest.items()
        if item.verdict in ("fail", "error")
    }
    blocking |= {
        test_id
        for test_id in required
        if test_id not in latest or latest[test_id].verdict != "pass"
    }
    tally = Counter(item.verdict for item in latest.values())
    counts = tuple((v, tally[v]) for v in VERDICTS)
    return Gate(not blocking, counts, tuple(sorted(blocking)))
