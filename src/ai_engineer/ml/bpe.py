"""Chapter 20: byte-pair encoding, trained and applied.

The tokenizer starts from the 256 possible byte values, so it can encode
any string, and learns merges: the most frequent adjacent pair of tokens
becomes a new token, again and again. Production tokenizers add
splitting rules, special tokens and speed; the core is this loop.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property

Pair = tuple[int, int]
Word = tuple[int, ...]

# Words keep their leading space; digits and punctuation split apart.
# Every character matches one branch, so splitting never loses text.
PIECES = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+")


def merge(word: Word, pair: Pair, new: int) -> Word:
    """Replace every occurrence of ``pair`` in ``word`` with ``new``."""
    out: list[int] = []
    i = 0
    while i < len(word):
        if word[i : i + 2] == pair:
            out.append(new)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


@dataclass(frozen=True)
class BPETokenizer:
    """Merges in the order they were learned; merge i makes token 256 + i."""

    merges: tuple[Pair, ...]

    @classmethod
    def train(cls, corpus: str, vocab_size: int) -> BPETokenizer:
        """Learn ``vocab_size - 256`` merges from ``corpus``."""
        words = Counter(tuple(p.encode()) for p in PIECES.findall(corpus))
        merges: list[Pair] = []
        while 256 + len(merges) < vocab_size:
            pairs: Counter[Pair] = Counter()
            for word, freq in words.items():
                for pair in itertools.pairwise(word):
                    pairs[pair] += freq
            if not pairs:
                break  # every word is already a single token
            best = min(pairs, key=lambda p: (-pairs[p], p))
            new = 256 + len(merges)
            words = Counter({merge(w, best, new): f for w, f in words.items()})
            merges.append(best)
        return cls(tuple(merges))

    @cached_property
    def ranks(self) -> dict[Pair, int]:
        return {pair: i for i, pair in enumerate(self.merges)}

    @cached_property
    def vocab(self) -> dict[int, bytes]:
        table = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            table[256 + i] = table[a] + table[b]
        return table

    def encode(self, text: str) -> list[int]:
        """Apply the learned merges, earliest first, to each piece."""
        ids: list[int] = []
        for piece in PIECES.findall(text):
            word: Word = tuple(piece.encode())
            while len(word) > 1:
                pairs = set(itertools.pairwise(word))
                best = min(
                    pairs, key=lambda p: self.ranks.get(p, len(self.merges))
                )
                if best not in self.ranks:
                    break
                word = merge(word, best, 256 + self.ranks[best])
            ids.extend(word)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")


def bytes_per_token(tokenizer: BPETokenizer, text: str) -> float:
    """UTF-8 bytes in ``text`` per token it encodes to; higher is cheaper."""
    ids = tokenizer.encode(text)
    if not ids:
        raise ValueError("cannot measure an empty text")
    return len(text.encode()) / len(ids)
