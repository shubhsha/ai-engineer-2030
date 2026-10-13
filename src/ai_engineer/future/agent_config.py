"""Chapter 69: a linter for the ``.claude`` directory of a repository.

A coding agent is shared infrastructure when its memory, permissions, hooks,
skills and subagents are files in the repository, reviewed like code. This
module reads those files and reports mistakes a reviewer would otherwise
have to spot by eye. It only reads files: it never runs a hook, a skill or
a command, and it never prints file contents, only rule names and paths.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# region finding
@dataclass(frozen=True, slots=True)
class Finding:
    """One problem: which rule fired, on which file, and what to do."""

    rule: str
    path: str
    message: str


# endregion finding

SETTINGS = ".claude/settings.json"
MEMORY = "CLAUDE.md"
# Substrings that make an allow rule too wide for a shared settings file.
DANGEROUS = (
    "git push",
    "git reset --hard",
    "rm -rf",
    "sudo",
    "chmod 777",
    "| sh",
    "| bash",
    "--force",
    "--no-verify",
)
# A bare wildcard after one of these verbs allows every subcommand.
WIDE_VERB = re.compile(r"Bash\((?:git|gh|docker|kubectl|npm|pip)[ :]\*\)")
HOOK_FILE = re.compile(r"[^\s\"']+\.(?:py|sh|js|ts)\b")
PROJECT_DIR = "${CLAUDE_PROJECT_DIR}"


def _strings(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


# region lint-settings
def lint_settings(
    settings: Mapping[str, Any], path: str = SETTINGS
) -> tuple[Finding, ...]:
    """Permission rules that are too wide, and no deny rule for secrets."""
    found: list[Finding] = []
    permissions = settings.get("permissions", {})
    if permissions.get("defaultMode") == "bypassPermissions":
        found.append(
            Finding("bypass-mode", path, "defaultMode skips every prompt")
        )
    for rule in _strings(permissions.get("allow", [])):
        bare = rule.strip() in ("*", "Bash", "Bash(*)", "Bash(:*)")
        if bare:
            found.append(
                Finding("allow-everything", path, f"allow rule {rule!r}")
            )
        elif WIDE_VERB.fullmatch(rule.strip()) or any(
            bad in rule for bad in DANGEROUS
        ):
            found.append(
                Finding("allow-dangerous", path, f"allow rule {rule!r}")
            )
    denied = " ".join(_strings(permissions.get("deny", [])))
    if ".env" not in denied:
        found.append(
            Finding("no-secret-deny", path, "no deny rule covers .env files")
        )
    return tuple(found)


# endregion lint-settings


def _read_settings(root: Path) -> Mapping[str, Any] | None:
    file = root / SETTINGS
    if not file.is_file():
        return None
    data = json.loads(file.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def _hook_commands(settings: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    for groups in settings.get("hooks", {}).values():
        for group in groups:
            yield from group.get("hooks", [])


# region lint-hooks
def lint_hooks(root: Path) -> tuple[Finding, ...]:
    """Hook commands that name a script that is not in the repository."""
    try:
        settings = _read_settings(root)
    except json.JSONDecodeError:
        return (Finding("settings-invalid", SETTINGS, "not valid JSON"),)
    found: list[Finding] = []
    for hook in _hook_commands(settings or {}):
        if hook.get("type") != "command":
            continue
        words = [*_strings(hook.get("command")), *_strings(hook.get("args"))]
        if not words:
            found.append(Finding("hook-empty", SETTINGS, "no command"))
        for match in HOOK_FILE.finditer(" ".join(words)):
            name = match.group(0).replace(PROJECT_DIR, str(root))
            target = Path(name)
            target = target if target.is_absolute() else root / target
            if not target.is_file():
                shown = match.group(0)
                found.append(
                    Finding("hook-missing-file", SETTINGS, f"no file {shown}")
                )
    return tuple(found)


# endregion lint-hooks


def frontmatter(text: str) -> dict[str, str]:
    """The ``key: value`` lines between the opening and closing ``---``."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    fields: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return fields
        key, colon, value = line.partition(":")
        if colon and not line.startswith((" ", "\t")):
            fields[key.strip()] = value.strip()
    return {}


def lint_skills(root: Path) -> tuple[Finding, ...]:
    """Each skill is a folder with a SKILL.md that names and describes it."""
    found: list[Finding] = []
    base = root / ".claude" / "skills"
    for folder in sorted(p for p in base.glob("*") if p.is_dir()):
        file = folder / "SKILL.md"
        shown = f".claude/skills/{folder.name}/SKILL.md"
        if not file.is_file():
            found.append(Finding("skill-no-file", shown, "no SKILL.md"))
            continue
        fields = frontmatter(file.read_text(encoding="utf-8"))
        for key in ("name", "description"):
            if not fields.get(key):
                found.append(
                    Finding(f"skill-no-{key}", shown, f"no {key} in header")
                )
    return tuple(found)


def lint_agents(root: Path) -> tuple[Finding, ...]:
    """Each subagent needs a name and a description to be delegated to."""
    found: list[Finding] = []
    base = root / ".claude" / "agents"
    for file in sorted(base.glob("*.md")):
        shown = f".claude/agents/{file.name}"
        fields = frontmatter(file.read_text(encoding="utf-8"))
        for key in ("name", "description"):
            if not fields.get(key):
                found.append(
                    Finding(f"agent-no-{key}", shown, f"no {key} in header")
                )
        if "tools" not in fields:
            found.append(
                Finding(
                    "agent-all-tools", shown, "tools omitted: inherits all"
                )
            )
    return tuple(found)


def memory_tokens(text: str) -> int:
    """An estimate: about four characters per token, rounded up.

    A real tokenizer gives a different count; the estimate is for a budget
    check, not for billing.
    """
    return math.ceil(len(text) / 4)


# region lint-project
def lint_project(root: Path, budget: int) -> tuple[Finding, ...]:
    """Every rule at once, plus the size of CLAUDE.md against a budget."""
    found: list[Finding] = []
    try:
        settings = _read_settings(root)
    except json.JSONDecodeError:
        settings = None
    if settings is not None:
        found.extend(lint_settings(settings))
    found.extend(lint_hooks(root))
    found.extend(lint_skills(root))
    found.extend(lint_agents(root))
    memory = root / MEMORY
    if memory.is_file():
        size = memory_tokens(memory.read_text(encoding="utf-8"))
        if size > budget:
            found.append(
                Finding("memory-over-budget", MEMORY, f"{size} > {budget}")
            )
    return tuple(found)


# endregion lint-project
