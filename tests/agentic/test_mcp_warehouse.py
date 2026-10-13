import asyncio

import pytest

from ai_engineer.mcp_server.warehouse import (
    QueryRejected,
    create_demo_db,
    describe_table,
    list_tables,
    run_query,
)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "shop.db"
    create_demo_db(path)
    return path


def orders(db):
    return run_query(db, "SELECT COUNT(*) FROM orders")["rows"][0][0]


def test_the_demo_database_is_small_and_synthetic(db):
    assert list_tables(db) == ["customers", "orders", "products"]
    assert orders(db) == 500


def test_describe_table_explains_columns_and_mistakes(db):
    columns = describe_table(db, "products")
    assert columns[1] == {"name": "name", "type": "TEXT", "nullable": False}
    with pytest.raises(QueryRejected, match="Call list_tables"):
        describe_table(db, "invoices")


@pytest.mark.parametrize(
    ("sql", "message"),
    [
        ("DELETE FROM orders", "Only a single read-only SELECT"),
        ("UPDATE orders SET quantity = 0", "Only a single read-only SELECT"),
        ("DROP TABLE orders", "Only a single read-only SELECT"),
        ("ATTACH DATABASE 'other.db' AS o", "Only a single read-only SELECT"),
        ("PRAGMA writable_schema = ON", "Only a single read-only SELECT"),
        ("WITH t AS (SELECT 1) DELETE FROM orders", "only reads"),
        ("SELECT 1; DELETE FROM orders", "one statement"),
        ("SELECT load_extension('x')", "only reads"),
        ("SELECT * FROM pragma_table_info('orders')", "only reads"),
    ],
)
def test_anything_but_a_read_is_refused(db, sql, message):
    with pytest.raises(QueryRejected, match=message):
        run_query(db, sql)
    assert orders(db) == 500
    assert not (db.parent / "other.db").exists()


def test_a_runaway_query_is_stopped(db):
    forever = (
        "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) "
        "SELECT COUNT(*) FROM n"
    )
    with pytest.raises(QueryRejected, match=r"longer than 0\.2 s"):
        run_query(db, forever, time_limit=0.2)


def test_results_are_capped_and_say_so(db):
    result = run_query(db, "SELECT id FROM orders ORDER BY id", max_rows=10)
    assert result["columns"] == ["id"]
    assert result["rows"] == [[i] for i in range(1, 11)]
    assert result["truncated"] is True


def test_a_missing_database_is_never_created(tmp_path):
    with pytest.raises(FileNotFoundError, match="no database"):
        list_tables(tmp_path / "missing.db")
    assert not (tmp_path / "missing.db").exists()


def test_the_server_over_the_protocol(db):
    pytest.importorskip("mcp")
    from mcp import Client

    from ai_engineer.mcp_server.warehouse import build_server

    async def session():
        async with Client(build_server(db)) as client:
            tools = (await client.list_tools()).tools
            count = await client.call_tool(
                "query", {"sql": "SELECT COUNT(*) AS n FROM customers"}
            )
            drop = await client.call_tool(
                "query", {"sql": "DROP TABLE orders"}
            )
        return tools, count, drop

    tools, count, drop = asyncio.run(session())
    assert sorted(t.name for t in tools) == [
        "describe_table",
        "list_tables",
        "query",
    ]
    assert all(t.annotations.read_only_hint for t in tools)
    assert count.structured_content == {
        "columns": ["n"],
        "rows": [[30]],
        "truncated": False,
    }
    assert drop.is_error
    assert "Only a single read-only SELECT" in drop.content[0].text
