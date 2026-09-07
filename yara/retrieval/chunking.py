"""Documents in, chunks out."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any

HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[\"'(])")


@dataclass
class Document:
    doc_id: str
    text: str
    title: str = ""
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_text(text: str, *, title: str = "", source: str = "", **metadata) -> "Document":
        doc_id = hashlib.sha256(f"{source}|{title}|{text}".encode()).hexdigest()[:16]
        return Document(doc_id, text, title, source, metadata)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    ordinal: int
    title: str = ""
    source: str = ""
    section: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Chunk":
        return Chunk(**data)

    def citation_label(self) -> str:
        parts = [p for p in (self.title or self.source, self.section) if p]
        return " > ".join(parts) or self.doc_id


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(text) if s.strip()]


def chunk_document(doc: Document, *, chunk_size: int = 900, overlap: int = 150) -> list[Chunk]:
    """Split on headings, then pack sentences up to chunk_size.

    Packing on sentence boundaries rather than a fixed window because the
    critic checks claims against one chunk at a time, and a chunk that ends
    mid-claim can't support the claim it half contains.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    for section, body in _sections(doc.text):
        for text in _pack(body, chunk_size, overlap):
            n = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}:{n:04d}",
                    doc_id=doc.doc_id,
                    text=text,
                    ordinal=n,
                    title=doc.title,
                    source=doc.source,
                    section=section,
                    metadata=dict(doc.metadata),
                )
            )
    return chunks


def _sections(text: str) -> list[tuple[str, str]]:
    headings = list(HEADING.finditer(text))
    if not headings:
        return [("", text)]

    out = []
    preamble = text[: headings[0].start()].strip()
    if preamble:
        out.append(("", preamble))

    for i, h in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        body = text[h.end() : end].strip()
        if body:
            out.append((h.group(2).strip(), body))
    return out


def _pack(text: str, chunk_size: int, overlap: int) -> list[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []

    out, current, size = [], [], 0

    for s in sentences:
        # One sentence longer than the budget: hard split rather than drop it.
        if len(s) > chunk_size:
            if current:
                out.append(" ".join(current))
                current, size = [], 0
            out += [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]
            continue

        if size + len(s) + 1 > chunk_size and current:
            out.append(" ".join(current))
            current, size = _tail(current, overlap)

        current.append(s)
        size += len(s) + 1

    if current:
        out.append(" ".join(current))
    return [c for c in out if c.strip()]


def _tail(sentences: list[str], budget: int) -> tuple[list[str], int]:
    """Trailing sentences to repeat at the head of the next chunk."""
    out, size = [], 0
    for s in reversed(sentences):
        if size + len(s) > budget:
            break
        out.insert(0, s)
        size += len(s) + 1
    return out, size
