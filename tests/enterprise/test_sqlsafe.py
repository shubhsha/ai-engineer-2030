import sqlite3

import pytest

from ai_engineer.enterprise.sqlsafe import UnsafeSql, guarded_query

ALLOWED = {"orders"}


@pytest.fixture
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:", isolation_level=None)
    c.executescript(
        "create table orders(id integer, amount real);"
        "create table salaries(who text, pay integer);"
        "insert into orders values (1, 50), (2, 150), (3, -5);"
        "insert into salaries values ('a', 1);"
    )
    return c


def test_a_select_on_an_allowed_table_runs(conn):
    result = guarded_query(conn, "SELECT sum(amount) FROM orders", ALLOWED)
    assert result.rows == ((195.0,),)
    assert not result.truncated


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE orders",
        "drop table orders",
        "DELETE FROM orders",
        "WITH x AS (SELECT 1) DELETE FROM orders",
        "ATTACH DATABASE 'evil.db' AS e",
        "PRAGMA writable_schema = 1",
        "CREATE TABLE loot AS SELECT * FROM orders",
        "SELECT 1; DROP TABLE orders",
        "SELECT * FROM salaries",
        "SELECT * FROM sqlite_master",
        "SELECT * FROM pragma_table_info('orders')",
        "SELECT load_extension('x')",
    ],
)
def test_everything_else_is_refused_and_nothing_changes(conn, sql):
    with pytest.raises(UnsafeSql):
        guarded_query(conn, sql, ALLOWED)
    assert conn.execute("select count(*) from orders").fetchone() == (3,)


def test_rows_are_capped_and_flagged(conn):
    result = guarded_query(conn, "SELECT id FROM orders", ALLOWED, max_rows=2)
    assert len(result.rows) == 2
    assert result.truncated


def test_a_runaway_query_hits_the_work_limit(conn):
    conn.executemany(
        "insert into orders values (?, 1)", [(i,) for i in range(500)]
    )
    sql = "SELECT count(*) FROM orders a, orders b, orders c"
    with pytest.raises(UnsafeSql, match="work limit"):
        guarded_query(conn, sql, ALLOWED, max_steps=100_000)


def test_the_guard_removes_itself_after_each_query(conn):
    with pytest.raises(UnsafeSql):
        guarded_query(conn, "DELETE FROM orders", ALLOWED)
    conn.execute("DELETE FROM salaries")  # the caller's own rights return
    assert conn.execute("select count(*) from salaries").fetchone() == (0,)
