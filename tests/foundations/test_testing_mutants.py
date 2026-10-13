"""Chapter 11: a mutation tester in a dozen lines, and what it finds.

A mutant is the code with one small change. If every check still passes,
the checks never looked at what the change affects.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable, Iterator
from typing import Any

from ai_engineer.foundations import commitmsg

Check = Callable[[str], list[str]]

PAIRS: list[tuple[type[ast.cmpop], type[ast.cmpop]]] = [
    (ast.Gt, ast.GtE),
    (ast.Lt, ast.LtE),
    (ast.Eq, ast.NotEq),
    (ast.Is, ast.IsNot),
    (ast.In, ast.NotIn),
]
SWAPS = dict(PAIRS) | {b: a for a, b in PAIRS}


# region mini-mutator
def sites(tree: ast.AST) -> list[ast.Compare]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare) and type(node.ops[0]) in SWAPS
    ]


def mutants(source: str) -> Iterator[tuple[int, str]]:
    """Yield (line, code), one copy of the code per comparison swapped."""
    for k in range(len(sites(ast.parse(source)))):
        tree = ast.parse(source)
        node = sites(tree)[k]
        node.ops[0] = SWAPS[type(node.ops[0])]()
        yield node.lineno, ast.unparse(tree)


# endregion


# region survivors
def load(code: str) -> Check:
    scope: dict[str, Any] = dict(vars(commitmsg))
    exec(code, scope)
    check: Check = scope["check_commit_message"]
    return check


def survivors(source: str, suite: Callable[[Check], None]) -> list[int]:
    """The lines whose mutant still passes: the suite never noticed."""
    alive = []
    for line, code in mutants(source):
        try:
            suite(load(code))
        except Exception:  # a failed assertion or a crash: noticed
            continue
        alive.append(line)
    return alive


# endregion

SOURCE = inspect.getsource(commitmsg.check_commit_message)
LIMIT_LINE = next(
    n
    for n, text in enumerate(SOURCE.splitlines(), 1)
    if "> MAX_HEADER" in text
)


def examples(check: Check) -> None:
    """What the repository's own tests check, in four lines."""
    assert check("fix: handle empty tool results\n\nThe loop crashed.") == []
    assert check("Fixed stuff") != []
    assert "limit is 72" in " ".join(check("fix: " + "x" * 80))
    assert "blank line" in " ".join(check("fix: add x\nbody right away"))


def with_the_edge(check: Check) -> None:
    examples(check)
    assert check("fix: " + "x" * 67) == []  # 72 characters: allowed
    assert check("fix: " + "x" * 68) != []  # 73 characters: too long


def test_the_mutator_finds_every_comparison() -> None:
    lines = [line for line, _ in mutants(SOURCE)]
    assert len(lines) == 4


def test_examples_alone_let_the_limit_mutant_live() -> None:
    assert survivors(SOURCE, examples) == [LIMIT_LINE]


def test_a_check_on_each_side_of_the_edge_kills_it() -> None:
    assert survivors(SOURCE, with_the_edge) == []
