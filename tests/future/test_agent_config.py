"""Chapter 69: the .claude linter finds each seeded mistake, and no more."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_engineer.future.agent_config import (
    Finding,
    frontmatter,
    lint_agents,
    lint_hooks,
    lint_project,
    lint_settings,
    lint_skills,
    memory_tokens,
)

REPO = Path(__file__).resolve().parents[2]
GOOD_SETTINGS = {
    "permissions": {
        "allow": ["Bash(make test)"],
        "deny": ["Read(./.env)"],
    }
}


def write(root: Path, name: str, text: str) -> None:
    file = root / name
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(text, encoding="utf-8")


def rules(findings: tuple[Finding, ...]) -> list[str]:
    return [f.rule for f in findings]


def good_project(root: Path) -> None:
    write(root, ".claude/settings.json", json.dumps(GOOD_SETTINGS))
    write(root, "CLAUDE.md", "Run make test.\n")
    write(
        root,
        ".claude/skills/a/SKILL.md",
        "---\nname: a\ndescription: Does a.\n---\nBody\n",
    )
    write(
        root,
        ".claude/agents/r.md",
        "---\nname: r\ndescription: Reviews.\ntools: Read\n---\nBody\n",
    )


def test_this_repository_has_no_findings() -> None:
    assert lint_project(REPO, budget=2_000) == ()


def test_good_fixture_has_no_findings(tmp_path: Path) -> None:
    good_project(tmp_path)
    assert lint_project(tmp_path, budget=100) == ()


@pytest.mark.parametrize(
    ("allow", "rule"),
    [
        ("Bash(git push *)", "allow-dangerous"),
        ("Bash(rm -rf *)", "allow-dangerous"),
        ("Bash(git *)", "allow-dangerous"),
        ("Bash(git:*)", "allow-dangerous"),
        ("Bash(*)", "allow-everything"),
        ("Bash", "allow-everything"),
    ],
)
def test_wide_allow_rules(allow: str, rule: str) -> None:
    settings = {"permissions": {"allow": [allow], "deny": ["Read(./.env)"]}}
    assert rules(lint_settings(settings)) == [rule]


def test_narrow_git_rules_pass() -> None:
    allow = ["Bash(git status *)", "Bash(git diff *)"]
    settings = {"permissions": {"allow": allow, "deny": ["Read(./.env)"]}}
    assert lint_settings(settings) == ()


def test_settings_without_secret_deny_and_bypass_mode() -> None:
    settings = {"permissions": {"defaultMode": "bypassPermissions"}}
    assert rules(lint_settings(settings)) == ["bypass-mode", "no-secret-deny"]


def test_hook_pointing_at_a_missing_file(tmp_path: Path) -> None:
    hook = {
        "type": "command",
        "command": "python3",
        "args": ["${CLAUDE_PROJECT_DIR}/.claude/hooks/gone.py"],
    }
    settings = {"hooks": {"PreToolUse": [{"hooks": [hook]}]}}
    write(tmp_path, ".claude/settings.json", json.dumps(settings))
    assert rules(lint_hooks(tmp_path)) == ["hook-missing-file"]
    write(tmp_path, ".claude/hooks/gone.py", "print('x')\n")
    assert lint_hooks(tmp_path) == ()


def test_invalid_settings_json(tmp_path: Path) -> None:
    write(tmp_path, ".claude/settings.json", "{oops")
    assert rules(lint_hooks(tmp_path)) == ["settings-invalid"]
    assert rules(lint_project(tmp_path, budget=10)) == ["settings-invalid"]


def test_skill_rules(tmp_path: Path) -> None:
    write(tmp_path, ".claude/skills/a/SKILL.md", "---\nname: a\n---\nBody\n")
    write(tmp_path, ".claude/skills/b/SKILL.md", "No header at all\n")
    (tmp_path / ".claude/skills/c").mkdir()
    assert rules(lint_skills(tmp_path)) == [
        "skill-no-description",
        "skill-no-name",
        "skill-no-description",
        "skill-no-file",
    ]


def test_agent_rules(tmp_path: Path) -> None:
    write(tmp_path, ".claude/agents/x.md", "---\nname: x\ntools: Read\n---\n")
    write(tmp_path, ".claude/agents/y.md", "---\ndescription: d\n---\n")
    assert rules(lint_agents(tmp_path)) == [
        "agent-no-description",
        "agent-no-name",
        "agent-all-tools",
    ]


def test_memory_over_budget(tmp_path: Path) -> None:
    good_project(tmp_path)
    write(tmp_path, "CLAUDE.md", "x" * 400)
    assert memory_tokens("x" * 400) == 100
    assert rules(lint_project(tmp_path, budget=99)) == ["memory-over-budget"]
    assert lint_project(tmp_path, budget=100) == ()


def test_frontmatter_stops_at_the_closing_line() -> None:
    text = "---\nname: a\n---\ndescription: not here\n"
    assert frontmatter(text) == {"name": "a"}
    assert frontmatter("---\nname: a\n") == {}
