"""Chapter 39: mark untrusted text, and check what answers cite."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from ai_engineer.llm.prompt import tagged


def untrusted(source: str, text: str) -> str:
    """Wrap outside text (a web page, an email, a tool result) so the
    system prompt can tell the model to treat it as data to analyze,
    never as instructions to follow. Delimiting lowers the risk of
    prompt injection; it does not remove it."""
    return tagged("untrusted", f"source: {source}\n{text}")


_CITATION = re.compile(r'\[(?P<source>[^\]\s]+)\]\s*"(?P<quote>[^"]+)"')


@dataclass(frozen=True, slots=True)
class CitationReport:
    supported: tuple[str, ...]
    unsupported: tuple[str, ...]
    unknown: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return bool(self.supported) and not (self.unsupported or self.unknown)


def _normal(text: str) -> str:
    return " ".join(text.lower().split())


def check_citations(answer: str, sources: Mapping[str, str]) -> CitationReport:
    """Check that every quote appears, word for word, in its source.

    Answers cite as ``[source-id] "exact words"``. An answer with no
    citations fails too: a claim without a source cannot be checked.
    """
    supported: list[str] = []
    unsupported: list[str] = []
    unknown: list[str] = []
    for m in _CITATION.finditer(answer):
        source, quote = m["source"], m["quote"]
        if source not in sources:
            unknown.append(source)
        elif _normal(quote) in _normal(sources[source]):
            supported.append(source)
        else:
            unsupported.append(f"{source}: {quote}")
    return CitationReport(tuple(supported), tuple(unsupported), tuple(unknown))
