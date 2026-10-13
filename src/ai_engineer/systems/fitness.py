"""Architecture fitness functions for chapter 58.

An architecture rule is only a rule if a machine can fail on it. This
module parses the source tree with ``ast`` (it never imports the code it
checks) and answers four questions: who imports whom, do the layers hold,
are optional dependencies kept out of module top level, and are the
decision records well formed.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

ROOT_PACKAGE = "ai_engineer"


@dataclass(frozen=True, slots=True)
class Violation:
    """One broken rule: where it broke and what was wrong."""

    where: str
    detail: str


# region import_graph
def _targets(tree: ast.AST, package: tuple[str, ...]) -> set[str]:
    """Top-level ai_engineer packages that one parsed module imports."""
    found: set[str] = set()
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - node.level + 1]
                names = [".".join((ROOT_PACKAGE, *base, node.module or ""))]
            elif node.module:
                names = [node.module]
        for name in names:
            parts = [p for p in name.split(".") if p]
            if parts[0] == ROOT_PACKAGE and len(parts) > 1:
                found.add(parts[1])
    return found


def import_graph(src_root: Path) -> dict[str, set[str]]:
    """Map each top-level package under ``src_root`` to those it imports."""
    graph: dict[str, set[str]] = {}
    for path in sorted(src_root.rglob("*.py")):
        rel = path.relative_to(src_root).with_suffix("")
        parts = rel.parts
        owner = parts[0] if len(parts) > 1 else "(top)"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        deps = graph.setdefault(owner, set())
        deps |= _targets(tree, parts[:-1])
        deps.discard(owner)
    return graph


# endregion


# region check_layers
def check_layers(
    graph: Mapping[str, set[str]],
    forbidden: Mapping[str, frozenset[str]],
) -> list[Violation]:
    """List each edge in ``graph`` that ``forbidden`` rules out.

    ``forbidden["foundations"] = {"claude"}`` means foundations may not
    import claude. A package missing from ``forbidden`` is unconstrained.
    """
    found: list[Violation] = []
    for package in sorted(forbidden):
        for target in sorted(graph.get(package, set()) & forbidden[package]):
            found.append(Violation(package, f"{package} imports {target}"))
    return found


# endregion


# region optional_imports
def _top_level_names(body: list[ast.stmt]) -> list[tuple[int, str]]:
    """Imports that run when the module loads (not inside a function)."""
    out: list[tuple[int, str]] = []
    for stmt in body:
        if isinstance(stmt, ast.Import):
            out += [(stmt.lineno, a.name.split(".")[0]) for a in stmt.names]
        elif isinstance(stmt, ast.ImportFrom) and stmt.module:
            if not stmt.level:
                out.append((stmt.lineno, stmt.module.split(".")[0]))
        elif isinstance(stmt, ast.If | ast.Try | ast.With):
            inner = [*stmt.body, *getattr(stmt, "orelse", [])]
            for handler in getattr(stmt, "handlers", []):
                inner += handler.body
            out += _top_level_names(inner)
    return out


def top_level_optional_imports(
    src_root: Path,
    names: frozenset[str],
    allowed: Mapping[str, frozenset[str]] | None = None,
) -> list[Violation]:
    """Modules that import an optional dependency when they load.

    ``allowed`` maps a dependency to the packages that own it, so
    ``{"fastapi": {"service"}}`` lets the service package import it at top
    level and nobody else.
    """
    owners = allowed or {}
    found: list[Violation] = []
    for path in sorted(src_root.rglob("*.py")):
        rel = path.relative_to(src_root)
        package = rel.parts[0] if len(rel.parts) > 1 else "(top)"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for line, name in _top_level_names(tree.body):
            if name in names and package not in owners.get(name, frozenset()):
                found.append(
                    Violation(f"{rel.as_posix()}:{line}", f"imports {name}")
                )
    return found


# endregion


# region lint_adr
REQUIRED_SECTIONS = (
    "Context",
    "Options",
    "Decision",
    "Because",
    "Consequences",
    "Revisit when",
)
_STATUS = re.compile(
    r"^- Status: (Proposed|Accepted|Rejected|Superseded by \[[^\]]+\]"
    r"\([^)]+\))\s*$",
    re.MULTILINE,
)


def lint_adr(text: str) -> list[str]:
    """Name what a decision record is missing (empty means it is whole)."""
    missing = [
        f"section: {name}"
        for name in REQUIRED_SECTIONS
        if not re.search(rf"^## {name}\s*$", text, re.MULTILINE)
    ]
    if not _STATUS.search(text):
        missing.append("status line")
    return missing


# endregion


# region superseded_links
_SUPERSEDED = re.compile(
    r"^- Status: Superseded by \[[^\]]+\]\(([^)]+)\)", re.MULTILINE
)


def superseded_links(adrs: Mapping[str, str]) -> list[Violation]:
    """Find "Superseded by" links that point at no record in ``adrs``."""
    found: list[Violation] = []
    for name in sorted(adrs):
        match = _SUPERSEDED.search(adrs[name])
        if match and match.group(1) not in adrs:
            found.append(Violation(name, f"missing {match.group(1)}"))
    return found


# endregion
