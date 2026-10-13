"""Chapter 42: an AI test engineer that proposes and never judges.

Seven steps: propose, validate, guard, approve, run, record, explain.
The model works in steps one and seven. Steps two to six are ordinary
code, so a verdict never depends on a model's mood.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from ai_engineer.enterprise.evidence import Evidence, Verdict
from ai_engineer.enterprise.sqlsafe import UnsafeSql, check_sql, guarded_query
from ai_engineer.enterprise.warehouse import Column
from ai_engineer.llm.model import Model, Request, Usage, user
from ai_engineer.llm.structured import generate_validated

KINDS = ("row_count", "null_count", "duplicate_keys", "total", "range")


# region test_proposal
@dataclass(frozen=True, slots=True)
class TestProposal:
    """One test the model suggests. It is data until a person approves it.

    ``sql`` must return one row with one column. ``expected`` is the text
    that column must equal for the test to pass.
    """

    __test__: ClassVar[bool] = False  # not a pytest class

    test_id: str
    kind: str
    scope: str
    sql: str
    expected: str
    rationale: str


# endregion test_proposal


PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["proposals"],
    "additionalProperties": False,
    "properties": {
        "proposals": {
            "type": "array",
            "maxItems": 20,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "test_id",
                    "kind",
                    "scope",
                    "sql",
                    "expected",
                    "rationale",
                ],
                "properties": {
                    "test_id": {"type": "string", "minLength": 3},
                    "kind": {"type": "string", "enum": list(KINDS)},
                    "scope": {"type": "string", "minLength": 3},
                    "sql": {"type": "string", "minLength": 6},
                    "expected": {"type": "string"},
                    "rationale": {"type": "string", "minLength": 10},
                },
            },
        }
    },
}


@dataclass(frozen=True, slots=True)
class Proposed:
    proposals: tuple[TestProposal, ...]
    attempts: int
    usage: Usage


def _distinct_ids(value: Any) -> list[str]:
    ids = [p["test_id"] for p in value["proposals"]]
    return [
        f"test_id {name!r} is used twice; ids must be unique"
        for name in sorted({i for i in ids if ids.count(i) > 1})
    ]


# region propose
def propose(
    model: Model,
    columns: Mapping[str, Sequence[Column]],
    *,
    max_attempts: int = 3,
) -> Proposed:
    """Ask the model for tests, repairing bad shapes in a loop.

    Only the metadata goes into the prompt: table and column names and
    types. No row values leave the warehouse.
    """
    prompt = (
        "Propose data-quality tests for these tables. Reply as JSON with "
        "a 'proposals' list. Each SQL must be one SELECT returning one "
        f"value.\nKinds: {', '.join(KINDS)}\n" + _describe(columns)
    )
    request = Request(
        system="You write read-only SQL tests. You never judge results.",
        messages=(user(prompt),),
        output_schema=PROPOSAL_SCHEMA,
    )
    got = generate_validated(
        model,
        request,
        PROPOSAL_SCHEMA,
        check=_distinct_ids,
        max_attempts=max_attempts,
    )
    items = tuple(TestProposal(**p) for p in got.value["proposals"])
    return Proposed(items, got.attempts, got.usage)


# endregion propose


def _describe(columns: Mapping[str, Sequence[Column]]) -> str:
    return "\n".join(
        f"{table}({', '.join(f'{c.name} {c.type}' for c in cols)})"
        for table, cols in columns.items()
    )


@dataclass(frozen=True, slots=True)
class Rejection:
    test_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class Screened:
    queued: tuple[TestProposal, ...]
    rejected: tuple[Rejection, ...]


# region screen
def screen(
    conn: sqlite3.Connection,
    proposals: Iterable[TestProposal],
    allowed_tables: Collection[str],
) -> Screened:
    """Compile each proposal under the guard. Nothing runs yet."""
    queued: list[TestProposal] = []
    rejected: list[Rejection] = []
    for item in proposals:
        try:
            check_sql(conn, item.sql, allowed_tables)
        except UnsafeSql as exc:
            rejected.append(Rejection(item.test_id, str(exc)))
        else:
            queued.append(item)
    return Screened(tuple(queued), tuple(rejected))


# endregion screen


def approve(
    queue: Iterable[TestProposal], approved_ids: Collection[str]
) -> tuple[TestProposal, ...]:
    """A person's decision, as data: only named tests move on."""
    return tuple(p for p in queue if p.test_id in approved_ids)


# region run_approved
def run_approved(
    conn: sqlite3.Connection,
    approved: Iterable[TestProposal],
    allowed_tables: Collection[str],
    *,
    run_id: str,
    now: Callable[[], float],
) -> list[Evidence]:
    """Run approved tests and record one Evidence each. No model here."""
    out: list[Evidence] = []
    for item in approved:
        verdict: Verdict = "error"
        observed = ""
        try:
            result = guarded_query(conn, item.sql, allowed_tables, max_rows=2)
            if len(result.rows) == 1 and len(result.rows[0]) == 1:
                observed = str(result.rows[0][0])
                verdict = "pass" if observed == item.expected else "fail"
            else:
                observed = "expected one row with one column"
        except UnsafeSql as exc:
            observed = str(exc)
        out.append(
            Evidence(
                item.test_id,
                run_id,
                now(),
                item.scope,
                item.expected,
                observed,
                verdict,
                location=item.sql,
            )
        )
    return out


# endregion run_approved


@dataclass(frozen=True, slots=True)
class Explanation:
    text: str
    cited_keys: tuple[str, ...]
    unknown_keys: tuple[str, ...]
    usage: Usage

    @property
    def verified(self) -> bool:
        return not self.unknown_keys


# region explain
def explain(
    model: Model,
    evidence: Evidence,
    sample_rows: Sequence[Mapping[str, Any]],
    key: str,
) -> Explanation:
    """Ask for a plain-words summary, then check every key it cites.

    The model sees the Evidence and a sample of offending rows. It must
    cite row keys. A key that is not in the sample marks the summary as
    unverified, and the dashboard shows it that way.
    """
    prompt = (
        f"Test {evidence.test_id} expected {evidence.expected!r} and saw "
        f"{evidence.observed!r}. Offending rows:\n"
        + json.dumps(list(sample_rows))
        + f"\nExplain for a business reader in two sentences. Cite the "
        f"'{key}' of each row you mention as JSON: "
        '{"summary": "...", "row_keys": ["..."]}'
    )
    schema: dict[str, Any] = {
        "type": "object",
        "required": ["summary", "row_keys"],
        "properties": {
            "summary": {"type": "string", "minLength": 10},
            "row_keys": {"type": "array", "items": {"type": "string"}},
        },
    }
    got = generate_validated(
        model, Request("You explain failures.", (user(prompt),)), schema
    )
    known = {str(r[key]) for r in sample_rows}
    cited = tuple(got.value["row_keys"])
    unknown = tuple(k for k in cited if k not in known)
    return Explanation(got.value["summary"], cited, unknown, got.usage)


# endregion explain
