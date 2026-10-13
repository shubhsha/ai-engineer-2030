"""The Claude Code hooks, run the way Claude Code runs them.

Each test sends a tool call as JSON on stdin and checks the exit code:
0 lets the call through, 2 blocks it (PreToolUse) or shows stderr to
Claude (PostToolUse).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).parents[2]
HOOKS = ROOT / ".claude" / "hooks"


def run_hook(
    name: str, file_path: str | Path, project: Path | None = None
) -> subprocess.CompletedProcess[str]:
    event: dict[str, Any] = {
        "hook_event_name": "PreToolUse",
        "tool_name": "Edit",
        "tool_input": {"file_path": str(file_path)},
        "cwd": str(project or Path.cwd()),
    }
    env = dict(os.environ)
    if project is not None:
        env["CLAUDE_PROJECT_DIR"] = str(project)
    return subprocess.run(
        [sys.executable, str(HOOKS / name)],
        input=json.dumps(event),
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )


# region protect-files
@pytest.mark.parametrize(
    "path",
    [
        "/work/repo/tests/foundations/legacy_pricing.py",
        "/work/repo/LICENSE",
        "/work/repo/.env",
        "/work/repo/deploy/.env.production",
        "/work/repo/certs/server.pem",
        "/work/repo/secrets/token.txt",
        r"C:\work\repo\tests\foundations\legacy_pricing.py",
    ],
)
def test_protected_paths_are_blocked_with_a_reason(path: str) -> None:
    result = run_hook("protect_files.py", path)
    assert result.returncode == 2
    assert "is protected:" in result.stderr


@pytest.mark.parametrize(
    "path",
    [
        "/work/repo/src/ai_engineer/foundations/pricing.py",
        "/work/repo/tests/foundations/test_pricing.py",
        "/work/repo/.env.example",
        "/work/repo/docs/secrets-handling.md",
    ],
)
def test_ordinary_paths_pass(path: str) -> None:
    result = run_hook("protect_files.py", path)
    assert result.returncode == 0
    assert result.stderr == ""


# endregion


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A project with its own ruff settings, so no user config leaks in."""
    if shutil.which("ruff", path=str(Path(sys.executable).parent)) is None:
        pytest.skip("ruff is not installed beside this Python")
    root = tmp_path / "project"
    root.mkdir()
    (root / "ruff.toml").write_text(
        'line-length = 79\n[lint]\nselect = ["E", "F", "I"]\n'
    )
    return root


# region ruff-hook
def test_fixable_problems_are_fixed_in_place(project: Path) -> None:
    module = project / "module.py"
    module.write_text("import os\nx=1\n")
    result = run_hook("ruff_after_edit.py", module, project)
    assert result.returncode == 0
    assert module.read_text() == "x = 1\n"


def test_remaining_problems_are_shown_to_claude(project: Path) -> None:
    module = project / "module.py"
    module.write_text("print(undefined_name)\n")
    result = run_hook("ruff_after_edit.py", module, project)
    assert result.returncode == 2
    assert "F821" in result.stderr


# endregion


def test_other_files_are_left_alone(project: Path, tmp_path: Path) -> None:
    notes = project / "notes.md"
    notes.write_text("x=1\n")
    outside = tmp_path / "outside.py"
    outside.write_text("x=1\n")
    for path in (notes, outside, project / "deleted.py"):
        assert run_hook("ruff_after_edit.py", path, project).returncode == 0
    assert notes.read_text() == "x=1\n"
    assert outside.read_text() == "x=1\n"


def test_settings_run_hooks_that_exist() -> None:
    settings = json.loads((ROOT / ".claude" / "settings.json").read_text())
    scripts = [
        arg.replace("${CLAUDE_PROJECT_DIR}", str(ROOT))
        for groups in settings["hooks"].values()
        for group in groups
        for handler in group["hooks"]
        for arg in handler.get("args", [])
    ]
    assert len(scripts) == 2
    assert all(Path(script).is_file() for script in scripts)
