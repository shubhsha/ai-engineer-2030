"""Chapter 29: split documents into overlapping, self-describing chunks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    doc_id: str
    text: str
    context: str

    @property
    def indexed(self) -> str:
        """What the index sees: the context header, then the text. A
        chunk that says only "the limit is 60" stays findable by the
        document and section it came from."""
        return f"{self.context}\n{self.text}"


def sections(text: str) -> list[tuple[str, str]]:
    """Split Markdown-style text into (heading, body) pairs."""
    found: list[tuple[str, str]] = []
    heading, lines = "", list[str]()
    for line in text.splitlines():
        if line.startswith("#"):
            if any(ln.strip() for ln in lines):
                found.append((heading, "\n".join(lines)))
            heading, lines = line.lstrip("#").strip(), []
        else:
            lines.append(line)
    if any(ln.strip() for ln in lines):
        found.append((heading, "\n".join(lines)))
    return found


def chunk_document(
    doc_id: str, title: str, text: str, *, size: int = 120, overlap: int = 20
) -> list[Chunk]:
    """Cut each section into windows of ``size`` words.

    Consecutive windows share ``overlap`` words, so a sentence cut at a
    boundary appears whole in one of them. Each chunk carries a header
    naming its document and section, a cheap, deterministic form of
    contextual retrieval.
    """
    if not 0 <= overlap < size:
        raise ValueError("overlap must be at least 0 and less than size")
    chunks: list[Chunk] = []
    for heading, body in sections(text):
        words = body.split()
        context = f"{title} > {heading}" if heading else title
        for start in range(0, max(len(words) - overlap, 1), size - overlap):
            piece = " ".join(words[start : start + size])
            chunks.append(
                Chunk(f"{doc_id}#{len(chunks)}", doc_id, piece, context)
            )
    return chunks
