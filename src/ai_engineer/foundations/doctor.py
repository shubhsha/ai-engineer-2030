"""Chapter 1: check that a workbench matches the book.

Each check is a small pure function that returns a :class:`Check`, so the
report is easy to test and easy to extend. Nothing here needs the network.
"""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

MIN_PYTHON = (3, 11)
Which = Callable[[str], str | None]


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def check_python(version: tuple[int, int] | None = None) -> Check:
    found = version or (sys.version_info.major, sys.version_info.minor)
    want = ".".join(map(str, MIN_PYTHON))
    have = ".".join(map(str, found))
    return Check("python", found >= MIN_PYTHON, f"{have} (need {want}+)")


def check_virtualenv(
    prefix: str | None = None, base: str | None = None
) -> Check:
    inside = (prefix or sys.prefix) != (base or sys.base_prefix)
    hint = "active" if inside else "inactive: source .venv/bin/activate"
    return Check("virtualenv", inside, hint)


def check_tool(
    name: str, which: Which = shutil.which, required: bool = True
) -> Check:
    path = which(name)
    detail = _tidy(path) if path else "not on PATH"
    return Check(name, path is not None, detail, required)


def _tidy(path: str) -> str:
    """Shorten the home directory to ``~`` so reports stay readable."""
    home = str(Path.home())
    return "~" + path[len(home) :] if path.startswith(home) else path


def check_file(path: Path, required: bool = True) -> Check:
    found = path.is_file()
    return Check(path.name, found, str(path) if found else "missing", required)


def check_api_key(env: Mapping[str, str] | None = None) -> Check:
    env = os.environ if env is None else env
    present = bool(env.get("ANTHROPIC_API_KEY"))
    detail = (
        "set: live examples enabled"
        if present
        else ("not set: offline mode, which is all the tests need")
    )
    return Check("api key", present, detail, required=False)


def check_environment(
    root: Path,
    env: Mapping[str, str] | None = None,
    which: Which = shutil.which,
) -> list[Check]:
    """Run every check. Optional checks warn; required checks fail."""
    return [
        check_python(),
        check_virtualenv(),
        check_tool("git", which),
        check_tool("ruff", which),
        check_tool("mypy", which),
        check_tool("pytest", which),
        check_tool("sqlite3", which, required=False),
        check_file(root / "pyproject.toml"),
        check_file(root / "configs" / "models.toml"),
        check_file(root / ".env", required=False),
        check_api_key(env),
    ]


def format_report(checks: Sequence[Check]) -> str:
    width = max((len(c.name) for c in checks), default=0)
    lines = []
    for c in checks:
        mark = "ok  " if c.ok else ("FAIL" if c.required else "warn")
        lines.append(f"{mark}  {c.name:<{width}}  {c.detail}")
    failed = sum(1 for c in checks if c.required and not c.ok)
    lines.append(
        "all required checks passed"
        if failed == 0
        else f"{failed} required check(s) failed"
    )
    return "\n".join(lines)


def exit_code(checks: Sequence[Check]) -> int:
    return 0 if all(c.ok or not c.required for c in checks) else 1
