import json
import sqlite3

import pytest

from ai_engineer.enterprise.evidence import release_gate
from ai_engineer.enterprise.test_engineer import (
    approve,
    explain,
    propose,
    run_approved,
    screen,
)
from ai_engineer.enterprise.warehouse import Column
from ai_engineer.llm.model import ScriptedModel
from ai_engineer.llm.structured import StructuredOutputError

COLUMNS = {
    "orders": [Column("id", "INTEGER", False), Column("amount", "REAL", True)]
}
ALLOWED = {"orders"}


def item(test_id, sql, expected="3", kind="row_count"):
    return {
        "test_id": test_id,
        "kind": kind,
        "scope": "bw.orders -> fabric.orders",
        "sql": sql,
        "expected": expected,
        "rationale": "a plain reason for the test",
    }


BATCH = [
    item("t_count", "SELECT count(*) FROM orders"),
    item("t_total", "SELECT sum(amount) FROM orders", "195.0", "total"),
    item(
        "t_dups",
        "SELECT count(*) - count(DISTINCT id) FROM orders",
        "0",
        "duplicate_keys",
    ),
    item(
        "t_range", "SELECT count(*) FROM orders WHERE amount < 0", "0", "range"
    ),
    item(
        "t_nulls",
        "SELECT count(*) FROM orders WHERE amount IS NULL",
        "0",
        "null_count",
    ),
    item("t_wipe", "DELETE FROM orders", "0"),
    item("t_leak", "SELECT count(*) FROM salaries", "1"),
]


def reply(proposals):
    return json.dumps({"proposals": proposals})


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:", isolation_level=None)
    c.executescript(
        "create table orders(id integer, amount real);"
        "create table salaries(who text, pay integer);"
        "insert into orders values (1, 50), (2, 150), (3, -5);"
    )
    return c


def test_a_bad_shape_is_repaired_with_a_reason(clock):
    bad = [dict(BATCH[0], kind="vibes")]
    model = ScriptedModel([reply(bad), reply(BATCH)])
    got = propose(model, COLUMNS)
    assert got.attempts == 2
    assert len(got.proposals) == 7
    repair = model.requests[1].messages[-1]["content"]
    assert "'vibes' is not one of" in repair


def test_duplicate_ids_are_sent_back_to_the_model():
    twice = [BATCH[0], dict(BATCH[1], test_id="t_count")]
    model = ScriptedModel([reply(twice), reply(BATCH)])
    assert propose(model, COLUMNS).attempts == 2


def test_the_model_cannot_loop_forever():
    model = ScriptedModel(default="not json")
    with pytest.raises(StructuredOutputError):
        propose(model, COLUMNS, max_attempts=2)


def test_the_prompt_carries_metadata_and_no_row_values():
    model = ScriptedModel([reply(BATCH)])
    propose(model, COLUMNS)
    sent = model.requests[0].messages[0]["content"]
    assert "orders(id INTEGER, amount REAL)" in sent


def test_the_guard_rejects_unsafe_proposals_before_anyone_reads_them(conn):
    model = ScriptedModel([reply(BATCH)])
    screened = screen(conn, propose(model, COLUMNS).proposals, ALLOWED)
    assert [r.test_id for r in screened.rejected] == ["t_wipe", "t_leak"]
    assert len(screened.queued) == 5
    assert conn.execute("select count(*) from orders").fetchone() == (3,)


def test_only_approved_tests_run_and_a_failure_is_evidence(conn, clock):
    model = ScriptedModel([reply(BATCH)])
    queue = screen(conn, propose(model, COLUMNS).proposals, ALLOWED).queued
    approved = approve(queue, {"t_count", "t_total", "t_range"})
    evidence = run_approved(
        conn, approved, ALLOWED, run_id="r1", now=clock.now
    )
    verdicts = {e.test_id: e.verdict for e in evidence}
    assert verdicts == {
        "t_count": "pass",
        "t_total": "pass",
        "t_range": "fail",
    }
    gate = release_gate(evidence, required=["t_count", "t_range"])
    assert not gate.passed
    assert gate.blocking == ("t_range",)


def test_a_test_that_cannot_run_is_an_error_not_a_pass(conn, clock):
    bad = item("t_two", "SELECT id, amount FROM orders")
    evidence = run_approved(
        conn, approve([_p(bad)], {"t_two"}), ALLOWED, run_id="r", now=clock.now
    )
    assert evidence[0].verdict == "error"


def _p(raw):
    from ai_engineer.enterprise.test_engineer import TestProposal

    return TestProposal(**raw)


def test_an_explanation_must_cite_real_row_keys(conn, clock):
    evidence = run_approved(
        conn, [_p(BATCH[3])], ALLOWED, run_id="r", now=clock.now
    )[0]
    rows = [{"id": 3, "amount": -5.0}]
    good = json.dumps(
        {"summary": "Order 3 has a negative amount.", "row_keys": ["3"]}
    )
    made_up = json.dumps(
        {"summary": "Orders 3 and 9 look wrong.", "row_keys": ["3", "9"]}
    )
    ok = explain(ScriptedModel([good]), evidence, rows, "id")
    bad = explain(ScriptedModel([made_up]), evidence, rows, "id")
    assert ok.verified
    assert not bad.verified
    assert bad.unknown_keys == ("9",)
