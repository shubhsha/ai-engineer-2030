"""Chapter 44: prove a change did only what it should.

A baseline is a stored copy of a report before a change. The manifest
says which cells may change and by what rule. ``compare`` sorts every
difference into expected, unexpected, missing or new, and the paired
test ``mcnemar_exact`` does the same job for two versions of an AI
system run on the same cases. Both write ``Evidence``.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

from ai_engineer.enterprise.evidence import Evidence

Key = tuple[Any, ...]
Normalizer = Callable[[Any], Any]
Kind = Literal["exact", "ratio", "within"]


def round_to(places: int) -> Normalizer:
    """Round numbers to ``places`` decimals so float noise cannot show."""
    quantum = Decimal(1).scaleb(-places)

    def apply(value: Any) -> Any:
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, (int, float, Decimal)):
            return Decimal(str(value)).quantize(quantum)
        return value

    return apply


# region snapshot
@dataclass(frozen=True, slots=True)
class Snapshot:
    """Rows keyed by a business key, with the noise already removed."""

    key_columns: tuple[str, ...]
    rows: Mapping[Key, Mapping[str, Any]]

    @classmethod
    def build(
        cls,
        rows: Iterable[Mapping[str, Any]],
        key: Sequence[str],
        *,
        ignore: Iterable[str] = (),
        normalizers: Mapping[str, Normalizer] | None = None,
    ) -> Snapshot:
        """Key the rows, drop ignored columns and normalize the rest.

        Row order never matters because rows are stored by key. A key
        that appears twice is an error: a baseline must be unambiguous.
        """
        skip = set(ignore) | set(key)
        fix = dict(normalizers or {})
        out: dict[Key, Mapping[str, Any]] = {}
        for row in rows:
            k = tuple(row[c] for c in key)
            if k in out:
                raise ValueError(f"duplicate business key {k!r}")
            out[k] = {
                c: fix[c](v) if c in fix else v
                for c, v in row.items()
                if c not in skip
            }
        return cls(tuple(key), out)

    def fingerprint(self) -> str:
        """A SHA-256 over the sorted content, to store instead of a copy."""
        digest = hashlib.sha256()
        for k in sorted(self.rows, key=repr):
            cells = sorted(self.rows[k].items())
            digest.update(repr((k, cells)).encode())
        return digest.hexdigest()


# endregion snapshot


# region manifest
@dataclass(frozen=True, slots=True)
class ChangeRule:
    """One permitted change: which cells, and what the new value may be.

    ``exact`` means the new value equals ``value``. ``ratio`` means the
    new value is the old one times ``value``. ``within`` means it moved
    by at most ``value``. ``tolerance`` widens ``exact`` and ``ratio``.
    ``where`` limits the rule to rows whose old values match.
    """

    column: str
    kind: Kind
    value: Decimal
    tolerance: Decimal = Decimal(0)
    where: tuple[tuple[str, Any], ...] = ()

    def covers(self, column: str, before: Mapping[str, Any]) -> bool:
        return column == self.column and all(
            before.get(c) == v for c, v in self.where
        )

    def target(self, before: Any) -> Decimal | None:
        """The value the rule predicts, or None for ``within``."""
        if self.kind == "exact":
            return self.value
        if self.kind == "ratio":
            return Decimal(str(before)) * self.value
        return None

    def accepts(self, before: Any, after: Any) -> bool:
        try:
            old, new = Decimal(str(before)), Decimal(str(after))
        except ArithmeticError:
            return False
        if self.kind == "within":
            return abs(new - old) <= self.value
        goal = self.target(old)
        assert goal is not None
        return abs(new - goal) <= self.tolerance


@dataclass(frozen=True, slots=True)
class Manifest:
    """The signed list of changes a release is allowed to make."""

    owner: str
    rules: tuple[ChangeRule, ...]
    allow_new_rows: bool = False


# endregion manifest


# region diff
@dataclass(frozen=True, slots=True)
class Finding:
    key: Key
    column: str
    before: Any
    after: Any


@dataclass(frozen=True, slots=True)
class Diff:
    """Every difference between two snapshots, sorted into four bins."""

    expected: tuple[Finding, ...]
    unexpected: tuple[Finding, ...]
    missing: tuple[Finding, ...]
    new: tuple[Finding, ...]
    plain_count: int
    allow_new_rows: bool = False

    @property
    def clean(self) -> bool:
        return not (
            self.unexpected
            or self.missing
            or (self.new and not self.allow_new_rows)
        )

    def to_evidence(
        self, *, run_id: str, recorded_at: float, scope: str
    ) -> Evidence:
        counts = (
            f"expected={len(self.expected)} "
            f"unexpected={len(self.unexpected)} "
            f"missing={len(self.missing)} new={len(self.new)}"
        )
        first = (self.unexpected or self.missing or self.new or (None,))[0]
        where = "" if first is None else f"{first.key!r}.{first.column}"
        return Evidence(
            test_id="progression.manifest",
            run_id=run_id,
            recorded_at=recorded_at,
            scope=scope,
            expected="only manifest changes; none missing",
            observed=counts,
            verdict="pass" if self.clean else "fail",
            location=where,
        )


def compare(before: Snapshot, after: Snapshot, manifest: Manifest) -> Diff:
    """Classify every cell that differs, and every cell that should."""
    expected: list[Finding] = []
    unexpected: list[Finding] = []
    missing: list[Finding] = []
    new: list[Finding] = []
    plain = 0
    for k in sorted(set(before.rows) | set(after.rows), key=repr):
        old, cur = before.rows.get(k), after.rows.get(k)
        if old is None:
            assert cur is not None
            new.append(Finding(k, "<row>", None, "added"))
            plain += 1
            continue
        if cur is None:
            unexpected.append(Finding(k, "<row>", "present", None))
            plain += 1
            continue
        for column in sorted(set(old) | set(cur)):
            a, b = old.get(column), cur.get(column)
            rule = next(
                (r for r in manifest.rules if r.covers(column, old)), None
            )
            changed = a != b
            plain += changed
            if rule is None:
                if changed:
                    unexpected.append(Finding(k, column, a, b))
            elif changed and rule.accepts(a, b):
                expected.append(Finding(k, column, a, b))
            elif changed:
                unexpected.append(Finding(k, column, a, b))
            elif rule.target(a) is None or not rule.accepts(a, b):
                missing.append(Finding(k, column, a, b))
    return Diff(
        tuple(expected),
        tuple(unexpected),
        tuple(missing),
        tuple(new),
        plain,
        manifest.allow_new_rows,
    )


# endregion diff


# region mcnemar
def discordant(
    a_passed: Sequence[bool], b_passed: Sequence[bool]
) -> tuple[int, int]:
    """Count cases only A passed and cases only B passed."""
    if len(a_passed) != len(b_passed):
        raise ValueError("both versions must run the same cases")
    pairs = list(zip(a_passed, b_passed, strict=True))
    return (
        sum(1 for a, b in pairs if a and not b),
        sum(1 for a, b in pairs if b and not a),
    )


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Two-sided exact p-value for the paired pass/fail comparison.

    Under the null hypothesis each discordant case is a fair coin flip
    between the two versions, so ``min(a_only, b_only)`` follows a
    binomial law with ``n = a_only + b_only`` and ``p = 1/2``.
    """
    if a_only < 0 or b_only < 0:
        raise ValueError("counts cannot be negative")
    n = a_only + b_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(a_only, b_only) + 1))
    return min(1.0, 2 * tail / (1 << n))


def paired_evidence(
    a_passed: Sequence[bool],
    b_passed: Sequence[bool],
    *,
    run_id: str,
    recorded_at: float,
    scope: str,
    alpha: float = 0.05,
) -> Evidence:
    """Fail when B is significantly worse; pass otherwise.

    Passing does not claim B is better: it says no regression was
    shown. Significant improvement is reported in ``observed``.
    """
    a_only, b_only = discordant(a_passed, b_passed)
    p = mcnemar_exact(a_only, b_only)
    worse = a_only > b_only and p < alpha
    better = b_only > a_only and p < alpha
    note = "B better" if better else "B worse" if worse else "no difference"
    return Evidence(
        test_id="progression.paired",
        run_id=run_id,
        recorded_at=recorded_at,
        scope=scope,
        expected=f"B not worse than A at alpha={alpha}",
        observed=f"a_only={a_only} b_only={b_only} p={p:.4f} ({note})",
        verdict="fail" if worse else "pass",
    )


# endregion mcnemar
