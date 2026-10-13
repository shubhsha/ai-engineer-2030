"""The code that the book prints must keep existing.

The book quotes this repository by name. ``docs/book-listings.txt``,
written by the book's build, lists every function, class, method and
``# region`` that a page prints, and every printed line range with a hash
of its text. A failure here means that a change would break a page: keep
the name, or say in the pull request which listing has to change.
"""

from __future__ import annotations

import ast
import hashlib
import re
from functools import cache
from pathlib import Path

ROOT = Path(__file__).parents[2]
MANIFEST = ROOT / "docs" / "book-listings.txt"
REGION = re.compile(r"#\s*region\s+(?P<name>[\w-]+)")
RANGE = re.compile(
    r"^(?P<path>\S+):(?P<a>\d+)-(?P<b>\d+) sha256:(?P<digest>[0-9a-f]+)$"
)
DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def entries() -> list[str]:
    lines = MANIFEST.read_text(encoding="utf-8").splitlines()
    return [line for line in lines if line and not line.startswith("#")]


@cache
def printable_names(path: Path) -> frozenset[str]:
    """Every name a listing can print from ``path``, as the build sees it."""
    text = path.read_text(encoding="utf-8")
    names = {m["name"] for m in REGION.finditer(text)}
    for node in ast.parse(text).body:
        if isinstance(node, DEFS):
            names.add(node.name)
        if isinstance(node, ast.ClassDef):
            names.update(
                f"{node.name}.{sub.name}"
                for sub in node.body
                if isinstance(sub, DEFS)
            )
    return frozenset(names)


def test_every_printed_name_still_exists() -> None:
    named = [entry.split("::") for entry in entries() if "::" in entry]
    assert len(named) > 40  # the manifest parsed; the book prints dozens
    missing = [
        f"{path}::{name}"
        for path, name in named
        if not (ROOT / path).is_file()
        or name not in printable_names(ROOT / path)
    ]
    assert missing == []


def test_every_printed_file_still_exists() -> None:
    files = [e for e in entries() if "::" not in e and not RANGE.match(e)]
    assert [f for f in files if not (ROOT / f).is_file()] == []


def test_printed_line_ranges_hold_the_same_text() -> None:
    changed = []
    for match in filter(None, map(RANGE.match, entries())):
        a, b = int(match["a"]), int(match["b"])
        text = (ROOT / match["path"]).read_text(encoding="utf-8")
        shown = "\n".join(text.split("\n")[a - 1 : b])
        digest = hashlib.sha256(shown.encode()).hexdigest()
        if not digest.startswith(match["digest"]):
            changed.append(f"{match['path']} lines {a}-{b}")
    assert changed == []
