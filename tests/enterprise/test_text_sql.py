import sqlite3
from decimal import Decimal

import pytest

from ai_engineer.enterprise.text_sql import (
    Case,
    MetricError,
    MetricRegistry,
    answer_with_evidence,
    generate_sql,
    result_match,
    schema_prompt,
    score,
    select_metric,
    suite_accuracy,
    tables_from,
)
from ai_engineer.llm.model import ScriptedModel

DDL = (
    "create table customers(customer_id integer primary key not null,"
    " region text);"
    "create table orders(order_id integer primary key not null,"
    " customer_id integer, order_date text not null,"
    " status text not null, freight real);"
    "create table order_lines(line_id integer primary key not null,"
    " order_id integer not null, qty integer not null,"
    " unit_price real not null, discount real);"
)


def make(extra_line: bool, null_discount: bool) -> sqlite3.Connection:
    c = sqlite3.connect(":memory:", isolation_level=None)
    c.executescript(DDL)
    c.execute("insert into customers values (1, 'EMEA'), (2, 'APAC')")
    c.execute("insert into orders values (1, 1, '2024-05-01', 'open', 10)")
    c.execute("insert into orders values (2, 2, '2024-06-01', 'open', 20)")
    c.execute("insert into order_lines values (1, 1, 2, 100, null)")
    c.execute("insert into order_lines values (2, 2, 1, 50, 0.5)")
    if extra_line:
        c.execute("insert into order_lines values (3, 1, 1, 10, 0)")
    if null_discount:
        c.execute("insert into order_lines values (4, 2, 1, 10, null)")
    return c


GOLD = "SELECT SUM(freight) FROM orders"
FANOUT = (
    "SELECT SUM(o.freight) FROM orders o JOIN order_lines l "
    "ON l.order_id = o.order_id"
)


def test_schema_prompt_marks_nullable_columns():
    text = schema_prompt(tables_from(make(False, False)))
    assert "customers(customer_id INTEGER, region TEXT?)" in text
    assert "discount REAL?" in text


def test_generate_sql_takes_the_fenced_block():
    model = ScriptedModel(["Sure.\n```sql\nSELECT 1;\n```"])
    assert generate_sql(model, "q", "t(a INTEGER)") == "SELECT 1"
    assert "t(a INTEGER)" in model.requests[0].system


def test_row_order_matters_only_when_the_gold_orders():
    c = make(False, False)
    a, b = "SELECT region FROM customers ORDER BY 1", "SELECT region "
    b += "FROM customers ORDER BY 1 DESC"
    assert not result_match(c, b, a)
    assert result_match(c, b.replace(" DESC", ""), a)
    assert result_match(
        c,
        "SELECT region FROM customers ORDER BY 1 DESC",
        "SELECT region FROM customers",
    )


def test_a_wrong_query_can_match_on_one_database_only():
    clean, busy = make(False, False), make(True, False)
    assert result_match(clean, FANOUT, GOLD)
    assert not result_match(busy, FANOUT, GOLD)
    assert not suite_accuracy([clean, busy], FANOUT, GOLD)
    assert suite_accuracy([clean, busy], GOLD, GOLD)


def test_score_separates_exact_execution_and_suite():
    clean, busy = make(False, False), make(True, False)
    reword = "select sum(freight) from orders"
    cases = [
        Case("a", GOLD, GOLD),
        Case("b", GOLD, FANOUT),
        Case("c", GOLD, reword.upper()),
    ]
    s = score(cases, [clean, busy])
    assert (s.cases, s.exact, s.execution, s.suite) == (3, 2, 3, 2)


def test_unsafe_or_broken_sql_never_matches():
    c = make(False, False)
    assert not result_match(c, "DROP TABLE orders", GOLD)
    assert not result_match(c, "SELECT nope FROM orders", GOLD)


def test_compose_writes_the_approved_sql():
    sql = MetricRegistry().compose(
        "net_revenue", {"fiscal_year": 2025}, ["region"]
    )
    assert "COALESCE(l.discount, 0)" in sql
    assert "o.status <> 'cancelled'" in sql
    assert sql.endswith("GROUP BY c.region ORDER BY c.region")


@pytest.mark.parametrize(
    ("metric", "filters", "group"),
    [
        ("nope", {}, ()),
        ("net_revenue", {"nope": 1}, ()),
        ("net_revenue", {}, ("nope",)),
        ("net_revenue", {"region": 1.5}, ()),
    ],
)
def test_compose_refuses_unknown_names(metric, filters, group):
    with pytest.raises(MetricError):
        MetricRegistry().compose(metric, filters, group)


def test_a_quote_in_a_filter_stays_a_string():
    sql = MetricRegistry().compose("order_count", {"region": "a' OR 1=1 --"})
    c = make(False, False)
    assert c.execute(sql).fetchone() == (0,)


def test_fan_out_cannot_happen_in_composed_sql():
    reg, busy = MetricRegistry(), make(True, False)
    rows = busy.execute(reg.compose("freight_total")).fetchall()
    assert rows == [(30.0,)]
    total = reg.compose("net_revenue")
    assert busy.execute(total).fetchone() == (200 + 25 + 10,)


def test_select_metric_composes_or_declines():
    reg = MetricRegistry()
    ask = '{"metric": "order_count", "group_by": ["region"]}'
    model = ScriptedModel(
        [ask, '{"metric": null}', "not json", '{"metric": "x"}']
    )
    first = select_metric(model, "q", reg)
    assert first is not None
    assert "COUNT(*)" in first
    assert select_metric(model, "q", reg) is None
    assert select_metric(model, "q", reg) is None
    assert select_metric(model, "q", reg) is None


def test_answer_is_withheld_when_the_total_does_not_reconcile():
    c, reg = make(False, False), MetricRegistry()
    sql = reg.compose("freight_total")
    kw = {"run_id": "r1", "recorded_at": 1.0}
    versions = reg.versions(["freight_total"])
    ok, ev = answer_with_evidence(
        c, "q", sql, versions, trusted=Decimal(30), **kw
    )
    assert ok is not None
    assert ev.verdict == "pass"
    assert ok.metrics == (("freight_total", 1),)
    bad, ev = answer_with_evidence(
        c, "q", sql, versions, trusted=Decimal(31), **kw
    )
    assert bad is None
    assert ev.verdict == "fail"
    free, ev = answer_with_evidence(c, "q", sql, versions, trusted=None, **kw)
    assert free is not None
    assert ev.verdict == "skip"
    none, ev = answer_with_evidence(
        c, "q", "DROP TABLE orders", (), trusted=None, **kw
    )
    assert none is None
    assert ev.verdict == "error"
