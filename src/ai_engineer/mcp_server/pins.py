"""Pin the tool definitions a person approved, and notice when they move.

A server controls its own tool descriptions, and a description is text
the model will read as if it were instructions. Approval is only worth
something if the host remembers *what* was approved. This module keeps
a short fingerprint per tool and reports any drift.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

_FIELDS = ("name", "description", "inputSchema", "annotations")


@dataclass(frozen=True, slots=True)
class Drift:
    """What changed between the approved tools and the tools now offered."""

    added: tuple[str, ...]
    removed: tuple[str, ...]
    changed: tuple[str, ...]

    @property
    def clean(self) -> bool:
        return not (self.added or self.removed or self.changed)


def fingerprint(tool: Mapping[str, Any]) -> str:
    """A stable 16-hex-digit hash of the parts the model reads."""
    kept = {k: tool.get(k) for k in _FIELDS}
    text = json.dumps(kept, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def pin(tools: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Fingerprint every tool by name, at the moment a person approves."""
    return {str(t["name"]): fingerprint(t) for t in tools}


def drift(
    approved: Mapping[str, str], tools: Iterable[Mapping[str, Any]]
) -> Drift:
    """Compare the approved pins with the tools a server offers now."""
    now = pin(tools)
    return Drift(
        added=tuple(sorted(now.keys() - approved.keys())),
        removed=tuple(sorted(approved.keys() - now.keys())),
        changed=tuple(
            sorted(
                n
                for n in now.keys() & approved.keys()
                if now[n] != approved[n]
            )
        ),
    )
