"""Chapter 46: JUnit XML to Evidence records.

JUnit XML is the interchange format almost every test runner can write,
though no body publishes a formal standard. The parser maps each test
case to one Evidence record and refuses documents that declare a DTD,
because entity expansion is how a small file becomes a large attack.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator

from ai_engineer.enterprise.evidence import Evidence, Verdict


class UnsafeXml(ValueError):
    """The document declares a DTD or an entity, so it is not parsed."""


_TAGS: tuple[tuple[str, Verdict], ...] = (
    ("error", "error"),
    ("failure", "fail"),
    ("skipped", "skip"),
)
_DECLARATION = re.compile(r"<!\s*(DOCTYPE|ENTITY)", re.IGNORECASE)


# region parse_junit
def parse_junit(
    xml_text: str, *, run_id: str, recorded_at: float, scope: str = "ci"
) -> tuple[Evidence, ...]:
    """One Evidence record per test case, with skip and error kept apart."""
    if _DECLARATION.search(xml_text):
        raise UnsafeXml("DTD and entity declarations are refused")
    root = ET.fromstring(xml_text)
    return tuple(
        _case_evidence(case, run_id, recorded_at, scope)
        for case in _cases(root)
    )


def _cases(root: ET.Element) -> Iterator[ET.Element]:
    yield from root.iter("testcase")


def _case_evidence(
    case: ET.Element, run_id: str, recorded_at: float, scope: str
) -> Evidence:
    classname = case.get("classname", "")
    name = case.get("name", "")
    test_id = f"{classname}.{name}" if classname else name
    verdict: Verdict = "pass"
    observed = "passed"
    for tag, kind in _TAGS:
        node = case.find(tag)
        if node is not None:
            verdict = kind
            observed = (node.get("message") or node.text or tag).strip()
            break
    return Evidence(
        test_id=test_id,
        run_id=run_id,
        recorded_at=recorded_at,
        scope=scope,
        expected="passed",
        observed=observed[:300],
        verdict=verdict,
        location=case.get("file", classname),
    )


# endregion
