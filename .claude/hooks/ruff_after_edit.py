#!/usr/bin/env python3
"""PostToolUse hook: format and lint each Python file that Claude edits.

Ruff fixes what it can in place. Whatever remains goes to stderr with
exit code 2, which Claude Code shows to Claude so that it can fix the
problem in its next step. Without ruff the hook does nothing.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_ruff(root: Path) -> str | None:
    """The project's ruff first, then one beside this Python, then PATH."""
    for candidate in (
        root / ".venv" / "bin" / "ruff",
        Path(sys.executable).parent / "ruff",
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return shutil.which("ruff")


def main() -> int:
    event = json.load(sys.stdin)
    file_path = event.get("tool_input", {}).get("file_path")
    if not isinstance(file_path, str) or not file_path.endswith(".py"):
        return 0
    root = Path(os.environ.get("CLAUDE_PROJECT_DIR") or event.get("cwd", "."))
    path = Path(file_path)
    inside = path.resolve().is_relative_to(root.resolve())
    if not inside or not path.is_file():
        return 0
    ruff = find_ruff(root)
    if ruff is None:
        # Exit 0: this goes to the debug log, not to Claude.
        print("ruff not found; run `make install`", file=sys.stderr)
        return 0
    # --force-exclude honors the project's excludes for explicit paths.
    common = ["--force-exclude", "--quiet", str(path)]
    # Fix first, then format, as ruff recommends; then report what is left.
    for step in (["check", "--fix"], ["format"]):
        subprocess.run(
            [ruff, *step, *common], cwd=root, capture_output=True, check=False
        )
    lint = subprocess.run(
        [ruff, "check", "--output-format=concise", *common],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if lint.returncode == 0:
        return 0
    print(lint.stdout + lint.stderr, end="", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
