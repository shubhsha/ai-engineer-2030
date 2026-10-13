"""Chapter 47: from thousands of Evidence records to one honest page.

``summarize`` counts, ``wilson`` and ``trend`` say how far a count can
be trusted, ``render_html`` prints a static page, and
``check_narrative`` refuses any sentence whose evidence ids are
missing or say the opposite.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from ai_engineer.enterprise.evidence import Evidence
from ai_engineer.evals.metrics import wilson_interval

DAY = 86_400.0


# region summary
@dataclass(frozen=True, slots=True)
class Summary:
    """One row of the dashboard: a scope or a day, with its denominator.

    ``n`` counts decided records (pass, fail, error). Skips are kept
    apart, so a rising pass rate cannot hide behind them.
    """

    key: str
    n: int
    passes: int
    fails: int
    errors: int
    skips: int
    last_at: float
    locations: tuple[str, ...]
    low: float
    high: float

    @property
    def rate(self) -> float:
        return self.passes / self.n if self.n else 0.0

    @property
    def skip_share(self) -> float:
        total = self.n + self.skips
        return self.skips / total if total else 0.0


# endregion summary


def wilson(passes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The 95 percent Wilson interval; with no data it is 0 to 1."""
    if n == 0:
        return 0.0, 1.0
    return wilson_interval(passes, n, z)


def _day(seconds: float) -> str:
    stamp = datetime.fromtimestamp(seconds, UTC)
    return stamp.strftime("%Y-%m-%d")


# region summarize
def summarize(
    evidence: Iterable[Evidence], by: Literal["scope", "day"] = "scope"
) -> tuple[Summary, ...]:
    """Group records by scope or by UTC day and count each verdict."""
    groups: dict[str, list[Evidence]] = {}
    for item in evidence:
        key = item.scope if by == "scope" else _day(item.recorded_at)
        groups.setdefault(key, []).append(item)
    rows = []
    for key in sorted(groups):
        items = groups[key]
        count = {v: sum(i.verdict == v for i in items) for v in _VERDICTS}
        n = count["pass"] + count["fail"] + count["error"]
        low, high = wilson(count["pass"], n)
        rows.append(
            Summary(
                key, n, count["pass"], count["fail"], count["error"],
                count["skip"], max(i.recorded_at for i in items),
                tuple(sorted({i.location for i in items if i.location})),
                low, high,
            )
        )  # fmt: skip
    return tuple(rows)


_VERDICTS = ("pass", "fail", "error", "skip")
# endregion summarize


# region trend
Direction = Literal["up", "down", "flat"]


def trend(rows: Sequence[Summary]) -> Direction:
    """Compare the first and last row; call it a change only when the
    two Wilson intervals do not overlap."""
    if len(rows) < 2:
        return "flat"
    first, last = rows[0], rows[-1]
    if last.low > first.high:
        return "up"
    if last.high < first.low:
        return "down"
    return "flat"


# endregion trend

_STATUS = {
    "ok": ("✔", "OK", "#1a7f4b"),
    "bad": ("✖", "BLOCKED", "#b3261e"),
    "stale": ("⧗", "STALE", "#8a6100"),
    "thin": ("?", "TOO FEW", "#5a5a5a"),
}


def _status(row: Summary, as_of: float, stale_after: float) -> str:
    if as_of - row.last_at > stale_after:
        return "stale"
    if row.fails or row.errors:
        return "bad"
    return "ok" if row.n >= 5 else "thin"


def _spark(rows: Sequence[Summary]) -> str:
    if len(rows) < 2:
        return ""
    w, h = 90, 24
    step = w / (len(rows) - 1)
    pts = " ".join(
        f"{i * step:.1f},{h - 2 - r.rate * (h - 4):.1f}"
        for i, r in enumerate(rows)
    )
    return (
        f'<svg width="{w}" height="{h}" role="img" '
        f'aria-label="pass rate by day"><polyline fill="none" '
        f'stroke="#444" stroke-width="1.5" points="{pts}"/></svg>'
    )


# region render
def render_html(
    summaries: Sequence[Summary],
    title: str,
    as_of: float,
    history: Mapping[str, Sequence[Summary]] | None = None,
    stale_after: float = DAY,
) -> str:
    """One static page: no scripts, every string escaped, every tile a
    link to its evidence, and an icon and a word beside every color."""
    history = history or {}
    tiles = []
    for row in summaries:
        icon, word, color = _STATUS[_status(row, as_of, stale_after)]
        href = html.escape(row.locations[0]) if row.locations else "#"
        lo, hi = row.low * 100, row.high * 100
        tiles.append(
            f'<section style="border-left:6px solid {color}">'
            f'<h2><a href="{href}">{html.escape(row.key)}</a></h2>'
            f'<p><b style="color:{color}">{icon} {word}</b> '
            f"{row.rate:.0%} of {row.n} (95% range {lo:.0f} to {hi:.0f}%)"
            f"; {row.skips} skipped ({row.skip_share:.0%} of all)</p>"
            f"{_spark(history.get(row.key, ()))}"
            f"<p>last evidence {html.escape(_day(row.last_at))}</p>"
            "</section>"
        )
    return (
        '<!doctype html><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title>"
        f"<h1>{html.escape(title)}</h1>"
        f"<p>As of {html.escape(_day(as_of))} UTC</p>" + "".join(tiles)
    )


# endregion render


# region narrative
@dataclass(frozen=True, slots=True)
class Problem:
    sentence: str
    reason: str


_CITE = re.compile(r"\[([^\]\s]+)\]")
_CAUSE = re.compile(r"\b(because|due to|caused by)\b", re.I)
_WORDS = {
    "pass": "pass", "passed": "pass", "fail": "fail", "failed": "fail",
    "failing": "fail", "error": "error", "errored": "error",
    "skipped": "skip",
}  # fmt: skip


def evidence_id(item: Evidence) -> str:
    return f"{item.test_id}@{item.run_id}"


def check_narrative(
    text: str, evidence: Iterable[Evidence]
) -> tuple[Problem, ...]:
    """Return the sentences that cite nothing, claim a cause, cite an
    unknown id or state a verdict that the cited record does not hold."""
    known = {evidence_id(e): e for e in evidence}
    problems = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        if not sentence:
            continue
        ids = _CITE.findall(sentence)
        if not ids:
            problems.append(Problem(sentence, "cites no evidence"))
            continue
        if _CAUSE.search(sentence):
            problems.append(Problem(sentence, "claims a cause"))
            continue
        missing = [i for i in ids if i not in known]
        if missing:
            problems.append(Problem(sentence, f"unknown id {missing[0]}"))
            continue
        claimed = {
            _WORDS[w]
            for w in re.findall(r"[a-z]+", _CITE.sub("", sentence.lower()))
            if w in _WORDS
        }
        if claimed and any(known[i].verdict not in claimed for i in ids):
            problems.append(Problem(sentence, "contradicts the record"))
    return tuple(problems)


# endregion narrative
