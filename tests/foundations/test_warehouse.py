import ast
from pathlib import Path

import pytest

from ai_engineer.enterprise.warehouse import (
    Column,
    InMemoryWarehouse,
    SqliteWarehouse,
    Warehouse,
)


@pytest.fixture
def wh() -> SqliteWarehouse:
    w = SqliteWarehouse()
    w.connection.executescript("""
        CREATE TABLE customer (id INTEGER PRIMARY KEY, name TEXT NOT NULL,
                               region TEXT);
        INSERT INTO customer VALUES (1, 'Ada', 'EU'), (2, 'Lin', NULL);
    """)
    return w


def test_adapter_satisfies_the_port(wh: SqliteWarehouse) -> None:
    port: Warehouse = wh  # checked statically by mypy, too
    assert port.tables() == ["customer"]


def test_columns_report_nullability(wh: SqliteWarehouse) -> None:
    assert wh.columns("customer") == [
        Column("id", "INTEGER", True),
        Column("name", "TEXT", False),
        Column("region", "TEXT", True),
    ]
    with pytest.raises(KeyError):
        wh.columns("missing")


def test_query_returns_plain_dicts(wh: SqliteWarehouse) -> None:
    rows = wh.query("SELECT name FROM customer WHERE region IS ?", ("EU",))
    assert rows == [{"name": "Ada"}]


def test_read_only_mode_refuses_writes(tmp_path: Path) -> None:
    path = tmp_path / "w.db"
    SqliteWarehouse(path).connection.execute("CREATE TABLE t (x INT)")
    ro = SqliteWarehouse(path, read_only=True)
    with pytest.raises(Exception, match="readonly"):
        ro.query("INSERT INTO t VALUES (1)")


# --- one contract, every adapter -----------------------------------------

CUSTOMER_COLUMNS = [
    Column("id", "INTEGER", True),
    Column("name", "TEXT", False),
    Column("region", "TEXT", True),
]
CUSTOMER_ROWS = [
    {"id": 1, "name": "Ada", "region": "EU"},
    {"id": 2, "name": "Lin", "region": None},
]


@pytest.fixture(params=["sqlite", "memory"])
def adapter(request: pytest.FixtureRequest) -> Warehouse:
    """The same two customers behind each adapter of the port."""
    if request.param == "sqlite":
        w = SqliteWarehouse()
        w.connection.executescript("""
            CREATE TABLE customer (id INTEGER PRIMARY KEY,
                                   name TEXT NOT NULL, region TEXT);
            INSERT INTO customer VALUES (1, 'Ada', 'EU'), (2, 'Lin', NULL);
        """)
        return w
    return InMemoryWarehouse(
        {"customer": CUSTOMER_COLUMNS}, {"customer": CUSTOMER_ROWS}
    )


def test_contract_lists_tables(adapter: Warehouse) -> None:
    assert adapter.tables() == ["customer"]


def test_contract_reports_columns(adapter: Warehouse) -> None:
    assert adapter.columns("customer") == CUSTOMER_COLUMNS
    with pytest.raises(KeyError):
        adapter.columns("missing")


def test_contract_select_all_returns_plain_dicts(adapter: Warehouse) -> None:
    assert adapter.query("SELECT * FROM customer") == CUSTOMER_ROWS


def test_the_fake_refuses_queries_it_cannot_run() -> None:
    fake = InMemoryWarehouse({"customer": CUSTOMER_COLUMNS}, {})
    with pytest.raises(NotImplementedError):
        fake.query("SELECT name FROM customer WHERE region = ?", ("EU",))


# --- architecture rules as tests -----------------------------------------

SRC = Path(__file__).resolve().parents[2] / "src" / "ai_engineer"


def _imports(path: Path) -> set[str]:
    """Dotted names that one source file imports (absolute imports)."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _package_graph() -> dict[str, set[str]]:
    """Which top-level package of ai_engineer imports which others."""
    graph: dict[str, set[str]] = {}
    for path in SRC.rglob("*.py"):
        parts = path.relative_to(SRC).parts
        owner = parts[0] if len(parts) > 1 else "(top)"
        deps = graph.setdefault(owner, set())
        for name in _imports(path):
            top = name.split(".")
            if top[0] == "ai_engineer" and len(top) > 1:
                deps.add(top[1])
        deps.discard(owner)
    return graph


def _find_cycle(graph: dict[str, set[str]]) -> list[str]:
    """One import cycle as a closed path of packages, or [] if none."""
    done: set[str] = set()

    def walk(node: str, path: list[str]) -> list[str]:
        if node in path:
            return [*path[path.index(node) :], node]
        if node in done:
            return []
        for nxt in sorted(graph.get(node, ())):
            found = walk(nxt, [*path, node])
            if found:
                return found
        done.add(node)
        return []

    for start in sorted(graph):
        found = walk(start, [])
        if found:
            return found
    return []


def test_cycle_finder_names_the_cycle() -> None:
    loop = {
        "report": {"pricing"},
        "pricing": {"catalog"},
        "catalog": {"report"},
    }
    assert _find_cycle(loop) == ["catalog", "report", "pricing", "catalog"]
    assert _find_cycle({"report": {"pricing"}, "pricing": set()}) == []


def test_packages_form_no_import_cycle() -> None:
    assert _find_cycle(_package_graph()) == []


def test_only_the_claude_adapter_imports_the_sdk() -> None:
    importers = {
        path.relative_to(SRC).as_posix()
        for path in SRC.rglob("*.py")
        if "anthropic" in {name.split(".")[0] for name in _imports(path)}
    }
    assert importers == {"claude/adapter.py"}
