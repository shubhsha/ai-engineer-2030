"""Chapter 40: relations catch faults that example tests walk past."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from decimal import Decimal
from typing import Any

from ai_engineer.enterprise.evidence import Evidence, Verdict, release_gate
from ai_engineer.enterprise.metamorphic import (
    CASE_AND_SPACE,
    MONTHS_ADD_UP,
    Relation,
    Sale,
    classify_ticket,
    classify_ticket_buggy,
    random_sales,
    revenue_by_month,
    revenue_by_month_buggy,
    run_relations,
)

TICKETS = ["Refund my invoice", "app crash on login", "thanks a lot"]


def examples_revenue(f: Callable[[tuple[Sale, ...]], object]) -> None:
    sales = (Sale(1, 15, Decimal(100)), Sale(3, 20, Decimal(70)))
    assert f(sales) == {1: Decimal(100), 3: Decimal(70)}


def examples_classifier(f: Callable[[str], str]) -> None:
    assert f("Refund my invoice") == "billing"
    assert f("app crash on login") == "bug"
    assert f("thanks") == "other"


def run(
    system: Callable[[Any], Any],
    relation: Relation[Any, Any],
    inputs: Sequence[Any],
) -> Evidence:
    return run_relations(
        system, [relation], inputs, run_id="r", recorded_at=0.0, scope="t"
    )[0]


def test_example_tests_pass_on_both_faulty_systems() -> None:
    examples_revenue(revenue_by_month_buggy)
    examples_classifier(classify_ticket_buggy)


def test_the_classifier_relation_catches_the_whitespace_fault() -> None:
    assert run(classify_ticket, CASE_AND_SPACE, TICKETS).verdict == "pass"
    bad = run(classify_ticket_buggy, CASE_AND_SPACE, TICKETS)
    assert bad.verdict == "fail"
    assert bad.test_id == "metamorphic.case_and_space_invariance"


def test_the_partition_relation_needs_an_input_on_the_edge() -> None:
    edge = (Sale(3, 1, Decimal(40)), Sale(2, 9, Decimal(10)))
    assert run(revenue_by_month, MONTHS_ADD_UP, [edge]).verdict == "pass"
    assert run(revenue_by_month_buggy, MONTHS_ADD_UP, [edge]).verdict == "fail"
    middle = (Sale(3, 15, Decimal(40)),)
    assert (
        run(revenue_by_month_buggy, MONTHS_ADD_UP, [middle]).verdict == "pass"
    )


def test_seeded_edge_inputs_find_the_fault_more_often() -> None:
    rng = random.Random(2030)

    def found(edges: bool) -> int:
        sets = [random_sales(rng, 20, edges=edges) for _ in range(200)]
        verdicts = [
            run(revenue_by_month_buggy, MONTHS_ADD_UP, [s]).verdict
            for s in sets
        ]
        return verdicts.count("fail")

    assert found(False) < 20 < found(True)


def test_a_relation_that_crashes_is_an_error_not_a_pass() -> None:
    boom: Relation[str, str] = Relation(
        "boom", str.upper, lambda *_: 1 // 0 > 1
    )
    assert run(classify_ticket, boom, TICKETS).verdict == "error"


def test_the_gate_story_in_the_chapter() -> None:
    def rec(test_id: str, verdict: Verdict) -> Evidence:
        return Evidence(test_id, "run-9", 1.0, "bw.sales", "ok", "-", verdict)

    records = [
        rec("count", "pass"),
        rec("sums", "pass"),
        rec("nulls", "pass"),
        rec("drift", "error"),
        rec("months_add_up", "pass"),
    ]
    required = ["count", "sums", "nulls", "months_add_up", "order_totals"]
    gate = release_gate(records, required=required)
    assert not gate.passed
    assert gate.blocking == ("drift", "order_totals")
    assert dict(gate.counts) == {"pass": 4, "fail": 0, "error": 1, "skip": 0}
