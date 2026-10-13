from __future__ import annotations

from pathlib import Path

from ai_engineer.systems.fitness import (
    check_layers,
    import_graph,
    lint_adr,
    superseded_links,
    top_level_optional_imports,
)

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "ai_engineer"
ADR_DIR = REPO / "docs" / "adr"

# region layer_rules
LEAVES = ("foundations", "llm")
FORBIDDEN = {
    "foundations": frozenset({"claude", "enterprise", "agents", "systems"}),
    "llm": frozenset({"claude", "enterprise", "agents", "foundations"}),
    "claude": frozenset({"enterprise", "agents", "service"}),
    "evals": frozenset({"enterprise", "agents", "service"}),
}
OPTIONAL = frozenset({"anthropic", "mcp", "fastapi"})
OWNERS = {
    "fastapi": frozenset({"service"}),
    "mcp": frozenset({"mcp_server"}),
}
# endregion


def _package(root: Path, name: str, body: str) -> None:
    (root / name).mkdir(parents=True)
    (root / name / "__init__.py").write_text("", encoding="utf-8")
    (root / name / "mod.py").write_text(body, encoding="utf-8")


# region test_real_tree_keeps_its_layers
def test_real_tree_keeps_its_layers() -> None:
    graph = import_graph(SRC)
    assert len(graph) >= 10
    assert check_layers(graph, FORBIDDEN) == []
    for leaf in LEAVES:
        assert graph[leaf] <= {"foundations"} - {leaf}


# endregion


def test_real_tree_imports_optional_packages_only_where_owned() -> None:
    found = top_level_optional_imports(SRC, OPTIONAL, OWNERS)
    assert found == []


# region test_planted_violation
def test_planted_layer_violation_is_caught(tmp_path: Path) -> None:
    _package(tmp_path, "foundations", "from ai_engineer.claude import cost\n")
    _package(tmp_path, "claude", "import ai_engineer.foundations\n")
    graph = import_graph(tmp_path)
    found = check_layers(graph, FORBIDDEN)
    assert [v.detail for v in found] == ["foundations imports claude"]


# endregion


def test_relative_imports_count_as_edges(tmp_path: Path) -> None:
    _package(tmp_path, "agents", "from .. import llm\nfrom ..llm import x\n")
    _package(tmp_path, "llm", "")
    assert import_graph(tmp_path)["agents"] == {"llm"}


def test_top_level_import_is_caught_and_lazy_import_is_not(
    tmp_path: Path,
) -> None:
    _package(tmp_path, "core", "import anthropic\n")
    _package(tmp_path, "lazy", "def f():\n    import anthropic\n")
    _package(
        tmp_path,
        "guarded",
        "try:\n    import mcp\nexcept ImportError:\n    mcp = None\n",
    )
    found = top_level_optional_imports(tmp_path, OPTIONAL, OWNERS)
    assert [(v.where, v.detail) for v in found] == [
        ("core/mod.py:1", "imports anthropic"),
        ("guarded/mod.py:2", "imports mcp"),
    ]


def test_owner_package_may_import_its_dependency(tmp_path: Path) -> None:
    _package(tmp_path, "service", "from fastapi import FastAPI\n")
    assert top_level_optional_imports(tmp_path, OPTIONAL, OWNERS) == []


# region test_adr_lint
def test_every_adr_in_the_repository_is_whole() -> None:
    names = sorted(p.name for p in ADR_DIR.glob("[0-9]*.md"))
    assert len(names) >= 4
    adrs = {n: (ADR_DIR / n).read_text(encoding="utf-8") for n in names}
    for name, text in adrs.items():
        assert lint_adr(text) == [], name
    assert superseded_links(adrs) == []


# endregion


def test_lint_names_what_is_missing() -> None:
    text = "# 0009. X\n\n- Status: maybe\n\n## Context\n\nx\n\n## Decision\n"
    assert lint_adr(text) == [
        "section: Options",
        "section: Because",
        "section: Consequences",
        "section: Revisit when",
        "status line",
    ]


def test_broken_supersession_link_is_reported() -> None:
    adrs = {
        "0001-a.md": "- Status: Superseded by [0002](0002-b.md)\n",
        "0003-c.md": "- Status: Superseded by [0009](0009-gone.md)\n",
        "0002-b.md": "- Status: Accepted\n",
    }
    found = superseded_links(adrs)
    assert [(v.where, v.detail) for v in found] == [
        ("0003-c.md", "missing 0009-gone.md")
    ]
