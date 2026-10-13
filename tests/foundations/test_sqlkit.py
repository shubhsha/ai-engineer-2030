import sqlite3
from pathlib import Path

import pytest

from ai_engineer.foundations.sqlkit import (
    NOT_EXISTS,
    NOT_IN,
    ORDER_COUNTS,
    ORDERS_SCHEMA,
    RENAME_REGION,
    adjust_stock,
    backfill_country,
    connect,
    query_plan,
    revenue_trend,
    run_migrations,
    schema_version,
    split_statements,
)

MIGRATIONS = [
    (1, "CREATE TABLE customer (id INTEGER PRIMARY KEY, name TEXT);"),
    (
        2,
        """
        CREATE TABLE daily_revenue (day TEXT PRIMARY KEY, revenue REAL);
        CREATE INDEX ix_revenue ON daily_revenue (revenue);
     """,
    ),
]


def test_migrations_apply_once_in_order() -> None:
    conn = connect()
    assert run_migrations(conn, MIGRATIONS) == [1, 2]
    assert run_migrations(conn, MIGRATIONS) == []
    assert schema_version(conn) == 2


def test_failed_migration_rolls_back_completely() -> None:
    conn = connect()
    run_migrations(conn, MIGRATIONS[:1])
    broken = (2, "CREATE TABLE t (x INT);\nINSERT INTO nope VALUES (1);")
    with pytest.raises(sqlite3.OperationalError):
        run_migrations(conn, [*MIGRATIONS[:1], broken])
    assert schema_version(conn) == 1
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert "t" not in tables  # DDL rolled back with the failed insert


def test_versions_must_ascend() -> None:
    with pytest.raises(ValueError, match="ascending"):
        run_migrations(connect(), [(2, "SELECT 1;"), (1, "SELECT 1;")])


def test_split_respects_semicolons_in_strings() -> None:
    script = "INSERT INTO t VALUES ('a;b');\nSELECT 1;\n"
    assert split_statements(script) == [
        "INSERT INTO t VALUES ('a;b');",
        "SELECT 1;",
    ]
    with pytest.raises(ValueError, match="incomplete"):
        split_statements("SELECT 1")


def test_window_functions_compute_running_totals() -> None:
    conn = connect()
    run_migrations(conn, MIGRATIONS)
    conn.executemany(
        "INSERT INTO daily_revenue VALUES (?, ?)",
        [("2026-10-01", 10.0), ("2026-10-02", 30.0), ("2026-10-03", 20.0)],
    )
    rows = revenue_trend(conn)
    assert [r["running_total"] for r in rows] == [10.0, 40.0, 60.0]
    assert [r["change"] for r in rows] == [None, 20.0, -10.0]
    assert rows[2]["avg_7d"] == pytest.approx(20.0)


SHOP = [(1, ORDERS_SCHEMA)]
SEED = """
INSERT INTO customer VALUES (1, 'Ada', 'EU');
INSERT INTO product VALUES (1, 'P1', 'Pen');
INSERT INTO product VALUES (2, 'P2', 'Pad');
INSERT INTO orders VALUES (1, 1, '2026-09-01');
INSERT INTO order_line VALUES (1, 1, 10, 200);
"""
SALES = [
    (1, 1, "EU", 10),
    (2, 2, "EU", 30),
    (3, 2, "US", 20),
    (4, 3, "EU", 30),
    (5, 4, "US", 40),
    (6, 5, "US", 20),
]
STOCK = "CREATE TABLE stock (sku TEXT PRIMARY KEY, qty INT CHECK (qty >= 0))"


def shop() -> sqlite3.Connection:
    conn = connect()
    run_migrations(conn, SHOP)
    return conn


def column(conn: sqlite3.Connection, sql: str) -> list[object]:
    return [row[0] for row in conn.execute(sql)]


def stock_db(path: Path | str, qty: int = 100) -> sqlite3.Connection:
    conn = connect(path)
    conn.execute(STOCK)
    conn.execute("INSERT INTO stock VALUES ('P1', ?)", (qty,))
    return conn


@pytest.mark.parametrize(
    ("statement", "message"),
    [
        ("INSERT INTO orders VALUES (2, 99, '2026-09-02')", "FOREIGN KEY"),
        ("INSERT INTO order_line VALUES (1, 2, 0, 500)", "CHECK"),
        ("INSERT INTO product VALUES (3, 'P1', 'Copy')", "UNIQUE"),
        ("INSERT INTO customer VALUES (2, NULL, 'EU')", "NOT NULL"),
        ("INSERT INTO order_line VALUES (1, 1, 5, 200)", "UNIQUE"),
    ],
)
def test_schema_rejects_what_the_model_forbids(
    statement: str, message: str
) -> None:
    conn = shop()
    conn.executescript(SEED)
    with pytest.raises(sqlite3.IntegrityError, match=message):
        conn.execute(statement)


def test_foreign_keys_are_per_connection(tmp_path: Path) -> None:
    path = tmp_path / "shop.db"
    conn = connect(path)
    run_migrations(conn, SHOP)
    orphan = "INSERT INTO orders VALUES (9, 99, '2026-09-09')"
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        conn.execute(orphan)
    forgetful = sqlite3.connect(path)  # never asked for the pragma
    forgetful.execute(orphan)
    forgetful.commit()
    check = forgetful.execute("PRAGMA foreign_key_check").fetchall()
    assert check == [("orders", 9, "customer", 0)]
    forgetful.close()
    conn.close()


def test_not_in_with_a_null_returns_nothing_but_not_exists_does_not() -> None:
    conn = shop()
    conn.execute("CREATE TABLE test_account (customer_id INTEGER)")
    customers = [(1, "Ada", "EU"), (2, "Ben", "US"), (3, "Cy", "EU")]
    conn.executemany("INSERT INTO customer VALUES (?, ?, ?)", customers)
    conn.execute("INSERT INTO test_account VALUES (2)")
    assert column(conn, NOT_IN) == column(conn, NOT_EXISTS) == ["Ada", "Cy"]
    conn.execute("INSERT INTO test_account VALUES (NULL)")
    assert column(conn, NOT_IN) == []
    assert column(conn, NOT_EXISTS) == ["Ada", "Cy"]


def test_left_join_counts_the_customer_with_no_orders_as_zero() -> None:
    conn = shop()
    names = ["Ada", "Ben", "Cy", "Di"]
    for number, name in enumerate(names, start=1):
        conn.execute(
            "INSERT INTO customer VALUES (?, ?, 'EU')", (number, name)
        )
    owners = [(1, 1), (2, 2), (3, 1), (4, 3)]
    conn.executemany("INSERT INTO orders VALUES (?, ?, '2026-09-01')", owners)
    counts = [(r["name"], r["orders"]) for r in conn.execute(ORDER_COUNTS)]
    assert counts == [("Ada", 2), ("Ben", 1), ("Cy", 1), ("Di", 0)]
    star = conn.execute(
        "SELECT COUNT(*) FROM customer c LEFT JOIN orders o "
        "ON o.customer_id = c.id WHERE c.name = 'Di'"
    ).fetchone()
    assert star[0] == 1  # COUNT(*) counts the padded row, COUNT(o.id) not


def test_range_frames_include_ties_and_ranks_differ() -> None:
    conn = connect()
    conn.execute(
        "CREATE TABLE sale "
        "(id INTEGER PRIMARY KEY, day INT, region TEXT, amount INT)"
    )
    conn.executemany("INSERT INTO sale VALUES (?, ?, ?, ?)", SALES)

    def over(expr: str) -> list[object]:
        return column(conn, f"SELECT {expr} FROM sale ORDER BY id")

    ties = [10, 60, 60, 90, 130, 150]  # the two day-2 rows are peers
    assert over("SUM(amount) OVER (ORDER BY day)") == ties
    rows = "ORDER BY day, id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
    assert over(f"SUM(amount) OVER ({rows})") == [10, 40, 60, 90, 130, 150]
    by_amount = "OVER (ORDER BY amount DESC)"
    assert over(f"RANK() {by_amount}") == [6, 2, 4, 2, 1, 4]
    assert over(f"DENSE_RANK() {by_amount}") == [4, 2, 3, 2, 1, 3]
    numbered = [6, 2, 4, 3, 1, 5]
    assert over("ROW_NUMBER() OVER (ORDER BY amount DESC, id)") == numbered


def test_adjust_stock_applies_deltas_and_the_check_stops_overselling() -> None:
    conn = stock_db(":memory:", qty=30)
    assert adjust_stock(conn, "P1", -20) == 10
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        adjust_stock(conn, "P1", -20)
    assert not conn.in_transaction
    assert column(conn, "SELECT qty FROM stock") == [10]
    with pytest.raises(KeyError):
        adjust_stock(conn, "nope", 1)
    assert not conn.in_transaction


def test_adjust_stock_fails_fast_when_another_writer_holds_the_lock(
    tmp_path: Path,
) -> None:
    seller_a = stock_db(tmp_path / "stock.db")
    seller_b = connect(tmp_path / "stock.db")
    seller_b.execute("PRAGMA busy_timeout = 0")
    seller_a.execute("BEGIN IMMEDIATE")  # A holds the write lock
    with pytest.raises(sqlite3.OperationalError) as busy:
        adjust_stock(seller_b, "P1", -20)
    assert busy.value.sqlite_errorname == "SQLITE_BUSY"
    assert not seller_b.in_transaction
    seller_a.execute("COMMIT")
    assert adjust_stock(seller_b, "P1", -20) == 80


def test_wal_reader_cannot_write_from_a_stale_snapshot(tmp_path: Path) -> None:
    seller_a = stock_db(tmp_path / "stock.db")
    seller_a.execute("PRAGMA journal_mode = WAL").fetchone()
    seller_b = connect(tmp_path / "stock.db")
    seller_b.execute("PRAGMA busy_timeout = 0")
    seller_b.execute("BEGIN")  # deferred: the first read starts the snapshot
    assert column(seller_b, "SELECT qty FROM stock") == [100]
    assert adjust_stock(seller_a, "P1", -10) == 90
    with pytest.raises(sqlite3.OperationalError) as stale:
        seller_b.execute("UPDATE stock SET qty = 80 WHERE sku = 'P1'")
    assert stale.value.sqlite_errorname == "SQLITE_BUSY_SNAPSHOT"
    seller_b.execute("ROLLBACK")
    assert adjust_stock(seller_b, "P1", -20) == 70


@pytest.mark.skipif(
    sqlite3.sqlite_version_info < (3, 53),
    reason="ALTER COLUMN DROP NOT NULL needs SQLite 3.53",
)
def test_rename_steps_keep_old_code_working_until_the_contract() -> None:
    conn = shop()
    conn.execute("INSERT INTO customer VALUES (1, 'Ada', 'EU')")
    new_code = (
        "INSERT INTO customer (id, name, country) VALUES (2, 'Ben', 'US')"
    )
    old_code = "INSERT INTO customer (id, name, region) VALUES (3, 'Cy', 'AP')"
    assert run_migrations(conn, [*SHOP, *RENAME_REGION[:1]]) == [3]
    with pytest.raises(sqlite3.IntegrityError, match=r"customer\.region"):
        conn.execute(new_code)  # the old NOT NULL still blocks new code
    assert run_migrations(conn, [*SHOP, *RENAME_REGION[:2]]) == [4]
    conn.execute(new_code)
    conn.execute(old_code)
    assert run_migrations(conn, [*SHOP, *RENAME_REGION[:3]]) == [5]
    countries = column(conn, "SELECT country FROM customer ORDER BY id")
    assert countries == ["EU", "US", "AP"]
    assert run_migrations(conn, [*SHOP, *RENAME_REGION]) == [6]
    with pytest.raises(sqlite3.OperationalError, match="no such column"):
        conn.execute("SELECT region FROM customer")
    countries = column(conn, "SELECT country FROM customer ORDER BY id")
    assert countries == ["EU", "US", "AP"]


def test_backfill_copies_in_batches_and_keeps_what_new_code_wrote() -> None:
    conn = shop()
    run_migrations(conn, [*SHOP, RENAME_REGION[0]])
    conn.executemany(
        "INSERT INTO customer (id, name, region) VALUES (?, ?, 'EU')",
        [(i, f"c{i}") for i in range(1, 8)],
    )
    conn.execute("UPDATE customer SET country = 'DE' WHERE id = 2")
    statements: list[str] = []
    conn.set_trace_callback(statements.append)
    assert backfill_country(conn, batch=3) == 6
    conn.set_trace_callback(None)
    assert sum(s.startswith("BEGIN") for s in statements) == 3
    countries = column(conn, "SELECT country FROM customer ORDER BY id")
    assert countries == ["EU", "DE", "EU", "EU", "EU", "EU", "EU"]
    assert backfill_country(conn, batch=3) == 0


def test_plans_show_the_index_and_the_leftmost_prefix_rule() -> None:
    conn = shop()
    by_customer = "SELECT * FROM orders WHERE customer_id = ?"
    by_day = "SELECT * FROM orders WHERE created_at >= ?"
    assert query_plan(conn, by_customer, (1,))[0].startswith("SCAN")
    conn.execute("CREATE INDEX ix_cust ON orders (customer_id, created_at)")
    seek = query_plan(conn, by_customer, (1,))[0]
    assert seek.startswith("SEARCH")
    assert "ix_cust" in seek
    assert query_plan(conn, by_day, ("2026-01-01",))[0].startswith("SCAN")
    conn.execute("CREATE INDEX ix_day ON orders (created_at)")
    wrapped = "SELECT id FROM orders WHERE date(created_at) = ?"
    ranged = "SELECT id FROM orders WHERE created_at >= ? AND created_at < ?"
    assert query_plan(conn, wrapped, ("2026-09-01",))[0].startswith("SCAN")
    days = ("2026-09-01", "2026-09-02")
    assert query_plan(conn, ranged, days)[0].startswith("SEARCH")
