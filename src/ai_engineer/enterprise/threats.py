"""Chapter 49: a threat model as data, enumerated with STRIDE.

Kohnfelder and Garg (1999) named six threat classes. Shostack (2014)
turned them into a per-element checklist: not every kind of element
can suffer every class. This module keeps that checklist as a table and
walks a data flow diagram with it, so the list of threats is a function
of the diagram and a change to the diagram changes the list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal["entity", "process", "store", "flow"]

STRIDE: dict[str, str] = {
    "S": "Spoofing",
    "T": "Tampering",
    "R": "Repudiation",
    "I": "Information disclosure",
    "D": "Denial of service",
    "E": "Elevation of privilege",
}

_APPLIES: dict[Kind, str] = {
    "entity": "SR",
    "process": "STRIDE",
    "store": "TID",
    "flow": "TID",
}


@dataclass(frozen=True, slots=True)
class Element:
    """One box, store or arrow on a data flow diagram.

    A flow names the elements it joins in ``src`` and ``dst``. A store
    that keeps logs can also suffer repudiation: someone denies an act
    and the log is the only witness.
    """

    name: str
    kind: Kind
    zone: str = ""
    src: str = ""
    dst: str = ""
    logs: bool = False

    def __post_init__(self) -> None:
        if self.kind == "flow" and not (self.src and self.dst):
            raise ValueError("a flow needs src and dst")
        if self.kind != "flow" and not self.zone:
            raise ValueError("a non-flow element needs a trust zone")


@dataclass(frozen=True, slots=True)
class Threat:
    element: str
    letter: str
    crosses_boundary: bool

    @property
    def category(self) -> str:
        return STRIDE[self.letter]


def stride_for(element: Element) -> str:
    """The STRIDE letters that apply to this kind of element."""
    letters = _APPLIES[element.kind]
    if element.kind == "store" and element.logs:
        letters = "TRID"
    return "".join(c for c in "STRIDE" if c in letters)


class ThreatModel:
    def __init__(self, elements: list[Element]) -> None:
        names = [e.name for e in elements]
        if len(set(names)) != len(names):
            raise ValueError("element names must be unique")
        self.elements = tuple(elements)
        self._zone = {e.name: e.zone for e in elements if e.kind != "flow"}
        for e in elements:
            if e.kind == "flow" and not {e.src, e.dst} <= set(self._zone):
                raise ValueError(f"flow {e.name} joins an unknown element")

    def _flows_of(self, name: str) -> list[Element]:
        return [
            e
            for e in self.elements
            if e.kind == "flow" and name in (e.src, e.dst)
        ]

    def crosses(self, element: Element) -> bool:
        """True if the element is a flow that leaves its zone, or sits
        at the end of one."""
        if element.kind == "flow":
            return self._zone[element.src] != self._zone[element.dst]
        return any(self.crosses(f) for f in self._flows_of(element.name))

    def threats(self) -> tuple[Threat, ...]:
        return tuple(
            Threat(e.name, letter, self.crosses(e))
            for e in self.elements
            for letter in stride_for(e)
        )

    def at_boundaries(self) -> tuple[Threat, ...]:
        """The threats worth reviewing first: those at a trust boundary."""
        return tuple(t for t in self.threats() if t.crosses_boundary)

    def to_table(self, *, boundary_only: bool = False) -> str:
        """A Markdown table, one row per element, letters in STRIDE order."""
        found = self.at_boundaries() if boundary_only else self.threats()
        rows: dict[str, str] = {}
        for t in found:
            rows[t.element] = rows.get(t.element, "") + t.letter
        lines = ["| Element | Kind | Zone | Threats |", "|---|---|---|---|"]
        for e in self.elements:
            if e.name in rows:
                zone = e.zone or f"{e.src} to {e.dst}"
                lines.append(
                    f"| {e.name} | {e.kind} | {zone} | {rows[e.name]} |"
                )
        return "\n".join(lines)


def rag_assistant() -> ThreatModel:
    """The data flow diagram of a RAG assistant over a document library."""
    return ThreatModel(
        [
            Element("Employee", "entity", "outside"),
            Element("Assistant API", "process", "app"),
            Element("Retriever", "process", "app"),
            Element("Vector index", "store", "app"),
            Element("Document library", "entity", "source"),
            Element("Model API", "entity", "vendor"),
            Element("Audit log", "store", "app", logs=True),
            Element("question", "flow", src="Employee", dst="Assistant API"),
            Element("query", "flow", src="Assistant API", dst="Retriever"),
            Element("lookup", "flow", src="Retriever", dst="Vector index"),
            Element(
                "ingest", "flow", src="Document library", dst="Vector index"
            ),
            Element("prompt", "flow", src="Assistant API", dst="Model API"),
            Element("record", "flow", src="Assistant API", dst="Audit log"),
        ]
    )
