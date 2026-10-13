"""Chapter 31: long-term memory with scope, provenance and expiry."""

from __future__ import annotations

from dataclasses import dataclass, field

from ai_engineer.foundations.abstractions import Clock, SystemClock
from ai_engineer.rag.bm25 import BM25


@dataclass(frozen=True, slots=True)
class Memory:
    scope: str  # whose memory this is, e.g. "user:42"
    key: str
    text: str
    source: str  # where it came from, so it can be checked or removed
    written_at: float
    expires_at: float | None


@dataclass
class MemoryStore:
    """Facts an agent may carry between sessions.

    Every fact has a scope, and reads never cross scopes; a source, so
    a wrong fact can be traced and corrected; and an optional expiry,
    because most of what an agent learns goes stale.
    """

    clock: Clock = field(default_factory=SystemClock)
    _facts: dict[tuple[str, str], Memory] = field(default_factory=dict)

    def write(
        self,
        scope: str,
        key: str,
        text: str,
        *,
        source: str,
        ttl: float | None = None,
    ) -> Memory:
        now = self.clock.now()
        expires = None if ttl is None else now + ttl
        memory = Memory(scope, key, text, source, now, expires)
        self._facts[scope, key] = memory  # a newer fact replaces an older
        return memory

    def forget(self, scope: str, key: str) -> bool:
        return self._facts.pop((scope, key), None) is not None

    def recall(self, scope: str, query: str, k: int = 3) -> list[Memory]:
        """The ``k`` live facts in ``scope`` that best match ``query``."""
        now = self.clock.now()
        live = [
            m
            for m in self._facts.values()
            if m.scope == scope
            and (m.expires_at is None or m.expires_at > now)
        ]
        hits = BM25([f"{m.key} {m.text}" for m in live]).search(query, k)
        return [live[i] for i, _ in hits]


class UntrustedSource(ValueError):
    """A fact was offered by a source that may not write memory."""


# region remember
def remember(
    store: MemoryStore,
    scope: str,
    key: str,
    text: str,
    *,
    source: str,
    trusted: frozenset[str],
    ttl: float | None = None,
) -> Memory:
    """Write ``text`` only if its source kind is on the trusted list.

    ``source`` reads ``kind:detail``, e.g. ``user:chat 2026-09-01``.
    Text that came from a web page or a tool result has another kind,
    so it can inform an answer but never become a standing fact.
    """
    kind = source.split(":", 1)[0]
    if kind not in trusted:
        raise UntrustedSource(source)
    return store.write(scope, key, text, source=source, ttl=ttl)


# endregion
