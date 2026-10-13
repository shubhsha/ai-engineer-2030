"""Chapter 39: find and mask secrets and personal data in text.

Run it on what goes into a prompt, what comes back from a tool and what
goes into a log. Patterns miss things and flag a few harmless strings,
so treat redaction as one layer, measured like any other classifier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PATTERNS: dict[str, re.Pattern[str]] = {
    "API_KEY": re.compile(
        r"\b(?:sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}"
        r"|gh[pousr]_[A-Za-z0-9]{36,})\b"
    ),
    "CARD": re.compile(r"\b\d(?:[ -]?\d){12,18}\b"),
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"),
    "PHONE": re.compile(r"(?<![\w+(])(?:\+|\()?\d[\d ().-]{7,}\d\b"),
}


def luhn_valid(number: str) -> bool:
    """The checksum every payment card number satisfies."""
    digits = [int(d) for d in number if d.isdigit()]
    total = 0
    for i, digit in enumerate(reversed(digits)):
        if i % 2:
            digit = digit * 2 - 9 if digit > 4 else digit * 2
        total += digit
    return 13 <= len(digits) <= 19 and total % 10 == 0


def _plausible(kind: str, text: str) -> bool:
    if kind == "CARD":
        return luhn_valid(text)
    if kind == "PHONE":  # E.164 allows at most 15 digits; dates have 8
        return 10 <= sum(c.isdigit() for c in text) <= 15
    return True


@dataclass(frozen=True, slots=True)
class Finding:
    kind: str
    start: int
    end: int


def find_sensitive(text: str) -> list[Finding]:
    """Matches in priority order; a span claimed once is not reused."""
    found: list[Finding] = []
    for kind, pattern in PATTERNS.items():
        for m in pattern.finditer(text):
            overlaps = any(
                m.start() < f.end and f.start < m.end() for f in found
            )
            if not overlaps and _plausible(kind, m.group()):
                found.append(Finding(kind, m.start(), m.end()))
    return sorted(found, key=lambda f: f.start)


@dataclass(frozen=True, slots=True)
class Redacted:
    text: str
    findings: tuple[Finding, ...]


def redact(text: str) -> Redacted:
    """Replace each finding with its kind, such as ``[EMAIL]``."""
    findings = find_sensitive(text)
    parts: list[str] = []
    last = 0
    for f in findings:
        parts += [text[last : f.start], f"[{f.kind}]"]
        last = f.end
    parts.append(text[last:])
    return Redacted("".join(parts), tuple(findings))
