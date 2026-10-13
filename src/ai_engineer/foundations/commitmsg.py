"""Chapter 7: a commit-msg hook that keeps history searchable.

``scripts/hooks/commit-msg`` runs this module, and ``make install``
enables it with ``git config core.hooksPath scripts/hooks``. CI runs
the same check on every commit in a pull request.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from pathlib import Path

TYPES = (
    "feat",
    "fix",
    "docs",
    "test",
    "refactor",
    "perf",
    "build",
    "ci",
    "chore",
    "revert",
)
HEADER = re.compile(
    r"^(?P<type>[a-z]+)(?:\((?P<scope>[a-z0-9./-]+)\))?"
    r"(?P<breaking>!)?: (?P<subject>\S.*)$"
)
MAX_HEADER = 72


def check_commit_message(message: str) -> list[str]:
    """Return every problem with a commit message; empty means accepted."""
    lines = [ln for ln in message.splitlines() if not ln.startswith("#")]
    if not lines or not lines[0].strip():
        return ["the message is empty"]
    header, problems = lines[0], []
    match = HEADER.match(header)
    if match is None:
        problems.append("header must read 'type(scope): subject'")
    elif match["type"] not in TYPES:
        problems.append(
            f"unknown type {match['type']!r}; use one of {', '.join(TYPES)}"
        )
    elif match["subject"].endswith("."):
        problems.append("subject should not end with a period")
    if len(header) > MAX_HEADER:
        problems.append(
            f"header has {len(header)} characters; the limit is {MAX_HEADER}"
        )
    if len(lines) > 1 and lines[1].strip():
        problems.append("leave a blank line after the header")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    message = Path(args[0]).read_text(encoding="utf-8")
    problems = check_commit_message(message)
    for problem in problems:
        print(f"commit-msg: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
