"""Chapter 34: hooks that decide in code what an agent may do.

A pre-hook sees each tool call before it runs and can allow it, deny it
or ask a person. A post-hook sees each result before the model does and
can rewrite it. Hooks run on every call, whatever the prompt says, which
is what makes them a control rather than a request.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Literal

from ai_engineer.guardrails.redact import redact
from ai_engineer.llm.model import JSON, ToolCall, tool_result
from ai_engineer.llm.tools import Executor, ToolRegistry

Action = Literal["allow", "ask", "deny"]
_SEVERITY: dict[Action, int] = {"allow": 0, "ask": 1, "deny": 2}


@dataclass(frozen=True, slots=True)
class Decision:
    action: Action
    reason: str = ""


PreHook = Callable[[ToolCall], Decision | None]
PostHook = Callable[[ToolCall, JSON], JSON]
Approver = Callable[[ToolCall, str], bool]

_SEGMENTS = re.compile(r"&&|\|\||[;|\n]")
_PREFIXES = {"sudo", "command", "nohup", "time", "xargs"}


def recursive_force_rm(command: str) -> bool:
    """True if any part of a shell command runs ``rm`` with both a
    recursive and a force flag, however the flags are spelled."""
    for segment in _SEGMENTS.split(command):
        try:
            words = shlex.split(segment)
        except ValueError:  # unbalanced quotes: fall back to spaces
            words = segment.split()
        while words and (words[0] in _PREFIXES or "=" in words[0]):
            words = words[1:]  # skip sudo, env assignments and the like
        if not words or PurePosixPath(words[0]).name != "rm":
            continue
        flags = [w for w in words[1:] if w.startswith("-")]
        short = "".join(f[1:] for f in flags if not f.startswith("--"))
        recursive = "--recursive" in flags or bool(set(short) & {"r", "R"})
        if recursive and ("--force" in flags or "f" in short):
            return True
    return False


DESTRUCTIVE: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\bDROP\s+(?:TABLE|DATABASE|SCHEMA)\b", re.I),
        "drops a table, database or schema",
    ),
    (re.compile(r"\bTRUNCATE\s+(?:TABLE\s+)?\w", re.I), "empties a table"),
    (
        re.compile(r"\bDELETE\s+FROM\s+\S+\s*(?:;|\Z)", re.I),
        "deletes every row (DELETE without WHERE)",
    ),
    (
        re.compile(r"\bgit\s+push\b[^;&|\n]*\s(?:--force|-f\b|\+\w)"),
        "force-pushes over shared history",
    ),
)


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def block_destructive(call: ToolCall) -> Decision | None:
    """Deny a call if any string in its input is a destructive command."""
    for text in _strings(call.input):
        if recursive_force_rm(text):
            return Decision("deny", "recursive forced delete (rm -rf)")
        for pattern, why in DESTRUCTIVE:
            if pattern.search(text):
                return Decision("deny", why)
    return None


def ask_before_writes(registry: ToolRegistry) -> PreHook:
    """Ask a person before any tool that is not marked read-only."""

    def hook(call: ToolCall) -> Decision | None:
        try:
            tool = registry[call.name]
        except KeyError:
            return None  # the registry reports unknown tools itself
        return None if tool.read_only else Decision("ask", "changes state")

    return hook


def redact_results(call: ToolCall, result: JSON) -> JSON:
    """Mask secrets and personal data before the model reads them."""
    content = result.get("content")
    if not isinstance(content, str):
        return result
    cleaned = redact(content)
    return {**result, "content": cleaned.text} if cleaned.findings else result


def _nobody(call: ToolCall, reason: str) -> bool:
    return False


def guarded(
    execute: Executor,
    *,
    pre: Sequence[PreHook] = (),
    post: Sequence[PostHook] = (),
    approve: Approver = _nobody,
    audit: list[JSON] | None = None,
) -> Executor:
    """Wrap an executor so that every call passes through the hooks.

    Every pre-hook runs and the strictest decision wins: deny over ask
    over allow. "ask" runs the call only if ``approve`` says yes; by
    default nobody does. Each decision is appended to ``audit``.
    """

    def run(call: ToolCall) -> JSON:
        decisions = [d for hook in pre if (d := hook(call)) is not None]
        verdict = max(
            decisions,
            key=lambda d: _SEVERITY[d.action],
            default=Decision("allow"),
        )
        approved = verdict.action == "ask" and approve(call, verdict.reason)
        if audit is not None:
            audit.append(
                {
                    "tool": call.name,
                    "call_id": call.id,
                    "decision": verdict.action,
                    "reason": verdict.reason,
                    "approved": approved,
                }
            )
        if verdict.action == "deny":
            message = f"Blocked by policy: {verdict.reason}. Do not retry."
            return tool_result(call.id, message, is_error=True)
        if verdict.action == "ask" and not approved:
            message = f"A person declined this action ({verdict.reason})."
            return tool_result(call.id, message, is_error=True)
        result = execute(call)
        for hook in post:
            result = hook(call, result)
        return result

    return run


def decision_counts(audit: Sequence[JSON]) -> dict[str, int]:
    """Count audit entries by decision, plus how many asks were approved."""
    counts = {"allow": 0, "ask": 0, "deny": 0, "approved": 0}
    for entry in audit:
        counts[entry["decision"]] += 1
        counts["approved"] += bool(entry["approved"])
    return counts
