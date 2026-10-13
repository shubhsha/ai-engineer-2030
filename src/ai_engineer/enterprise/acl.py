"""Chapter 49: access control inside retrieval.

A user must never see a chunk they could not open as a document. The
rule is simple to state and easy to break: if the index ranks first and
checks permission afterwards, the ranking already used chunks the user
may not see, and the result count gives them away. Here the permission
check runs before the ranking, and a missing ACL means nobody.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ai_engineer.rag.bm25 import BM25


@dataclass(frozen=True, slots=True)
class Principal:
    name: str
    groups: frozenset[str] = frozenset()

    @property
    def ids(self) -> frozenset[str]:
        return self.groups | {self.name}


@dataclass(frozen=True, slots=True)
class AclChunk:
    """Text plus the principals who may read the document it came from."""

    id: str
    text: str
    allowed: frozenset[str] = frozenset()


# region acl_authorized
def authorized(user: Principal, chunks: Sequence[AclChunk]) -> list[AclChunk]:
    """Chunks the user may read. Empty ACL means nobody (default deny)."""
    return [c for c in chunks if c.allowed & user.ids]


# endregion


class AclIndex:
    """BM25 over every chunk, plus the chunks' ACLs."""

    def __init__(self, chunks: Sequence[AclChunk]) -> None:
        self.chunks = tuple(chunks)
        self._bm25 = BM25([c.text for c in chunks])

    def search(self, query: str, k: int) -> list[AclChunk]:
        hits = self._bm25.search(query, k)
        return [self.chunks[i] for i, _ in hits]

    def score(self, query: str, i: int) -> float:
        return self._bm25.score(query, i)


# region retrieve_secure
def retrieve_secure(
    query: str, user: Principal, index: AclIndex, k: int
) -> list[AclChunk]:
    """Filter first, rank second: only chunks the user may read compete
    for the ``k`` places."""
    allowed = {c.id for c in authorized(user, index.chunks)}
    scored = [
        (i, index.score(query, i))
        for i, c in enumerate(index.chunks)
        if c.id in allowed
    ]
    hits = sorted((h for h in scored if h[1] > 0), key=lambda h: (-h[1], h[0]))
    return [index.chunks[i] for i, _ in hits[:k]]


# endregion


def retrieve_post_filter(
    query: str, user: Principal, index: AclIndex, k: int
) -> list[AclChunk]:
    """The tempting version: rank all chunks, then drop what the user
    may not read. Safe in the final list, wrong in two other ways."""
    return authorized(user, index.search(query, k))


def retrieve_unfiltered(
    query: str, user: Principal, index: AclIndex, k: int
) -> list[AclChunk]:
    """No access control at all: the baseline that leaks."""
    return index.search(query, k)
