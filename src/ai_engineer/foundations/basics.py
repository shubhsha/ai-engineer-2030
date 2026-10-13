"""Chapter 2: the core of the language, in small working functions."""

from __future__ import annotations

from collections import Counter


def parse_pairs(line: str, sep: str = ";") -> dict[str, str]:
    """Parse ``"user=ada; region=eu"`` into a dict.

    A fragment without ``=`` raises ValueError that quotes the fragment,
    so a bad log line tells you where it went wrong, not just that it did.
    """
    result: dict[str, str] = {}
    for raw in line.split(sep):
        fragment = raw.strip()
        if not fragment:
            continue
        key, eq, value = fragment.partition("=")
        if not eq or not key.strip():
            raise ValueError(f"expected key=value, got {fragment!r}")
        result[key.strip()] = value.strip()
    return result


def top_words(text: str, n: int = 3) -> list[tuple[str, int]]:
    """The ``n`` most common words, case-folded, punctuation stripped."""
    words = (w.strip(".,;:!?\"'()[]").casefold() for w in text.split())
    return Counter(w for w in words if w).most_common(n)


def add_tag_buggy(tag: str, tags: list[str] = []) -> list[str]:  # noqa: B006
    """The classic trap: the default list is created once and shared."""
    tags.append(tag)
    return tags


def add_tag(tag: str, tags: list[str] | None = None) -> list[str]:
    """Return a new list; never mutate the caller's, never share a default."""
    return [*(tags or []), tag]
