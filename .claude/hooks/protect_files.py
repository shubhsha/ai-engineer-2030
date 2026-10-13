#!/usr/bin/env python3
"""PreToolUse hook: refuse edits to files that must not change.

Claude Code sends each tool call as JSON on stdin. Exit code 2 blocks the
call, and the message on stderr tells Claude why. Matching is by path
suffix, so the rules hold in a git worktree and on Windows paths too.
"""

from __future__ import annotations

import fnmatch
import json
import sys
from pathlib import PurePosixPath

# Project paths that only a person changes, and why.
PROTECTED = {
    "tests/foundations/legacy_pricing.py": (
        "it is the characterization oracle for the pricing code; editing "
        "it would make the test compare the new code with itself"
    ),
    "LICENSE": "license terms are changed by a maintainer, not an agent",
}
# Secrets never live in the repository, at any depth.
SECRET_NAMES = (".env", ".env.*", "*.pem", "*.key")
NOT_SECRET = (".env.example",)
SECRET_DIRS = ("secrets",)


def reason_to_block(file_path: str) -> str | None:
    """Why ``file_path`` must not be edited, or None if it may be."""
    parts = PurePosixPath(file_path.replace("\\", "/")).parts
    for protected, reason in PROTECTED.items():
        tail = PurePosixPath(protected).parts
        if parts[-len(tail) :] == tail:
            return reason
    name = parts[-1] if parts else ""
    if any(part in SECRET_DIRS for part in parts[:-1]) or (
        name not in NOT_SECRET
        and any(fnmatch.fnmatchcase(name, p) for p in SECRET_NAMES)
    ):
        return "secrets never live in the repository; ask the user instead"
    return None


def main() -> int:
    event = json.load(sys.stdin)
    file_path = event.get("tool_input", {}).get("file_path")
    if not isinstance(file_path, str):
        return 0
    reason = reason_to_block(file_path)
    if reason is None:
        return 0
    print(f"Blocked: {file_path} is protected: {reason}.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
