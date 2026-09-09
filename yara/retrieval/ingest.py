"""Loading source material - files, directories, PDFs, web pages."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import httpx

from .chunking import Chunk, Document, chunk_document

logger = logging.getLogger("yara.retrieval")

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".py", ".json", ".csv", ".html"}


def load_text_file(path: Path) -> Document:
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".html":
        raw = strip_html(raw)
    return Document.from_text(raw, title=path.stem, source=str(path))


def load_pdf(path: Path) -> Document:
    try:
        import pypdf
    except ImportError as exc:
        raise ImportError("PDF loading needs pypdf. pip install pypdf") from exc

    reader = pypdf.PdfReader(str(path))
    text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
    return Document.from_text(
        text, title=Path(path).stem, source=str(path), pages=len(reader.pages)
    )


def load_url(url: str, timeout: float = 30.0) -> Document:
    r = httpx.get(url, timeout=timeout, follow_redirects=True)
    r.raise_for_status()

    body = r.text
    if "html" in r.headers.get("content-type", ""):
        body = strip_html(body)
    return Document.from_text(body, title=_title(r.text) or url, source=url)


def strip_html(html: str) -> str:
    """Crude but dependency-free. Drops nav/script/style, keeps block breaks.

    Not a real parser - it will mangle pathological markup. Good enough for
    article pages, which is what actually gets ingested.
    """
    html = re.sub(r"(?is)<(script|style|nav|footer|header)[^>]*>.*?</\1>", " ", html)
    html = re.sub(r"(?i)</(p|div|li|h[1-6]|tr)>", "\n", html)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;?", " ", text)
    text = re.sub(r"&amp;", "&", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _title(html: str) -> str | None:
    m = re.search(r"(?is)<title[^>]*>(.*?)</title>", html)
    return m.group(1).strip() if m else None


def load_path(path: Path) -> list[Document]:
    """One file, or everything supported under a directory."""
    path = Path(path)
    if path.is_file():
        return [_load_one(path)]
    if not path.is_dir():
        raise FileNotFoundError(path)

    docs = []
    for f in sorted(path.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in TEXT_SUFFIXES | {".pdf"}:
            continue
        try:
            docs.append(_load_one(f))
        except Exception as exc:
            # One unreadable file shouldn't abort ingesting a whole directory.
            logger.warning("skipping %s: %s", f, exc)
    return docs


def _load_one(path: Path) -> Document:
    if path.suffix.lower() == ".pdf":
        return load_pdf(path)
    return load_text_file(path)


def to_chunks(docs: list[Document], chunk_size: int = 900, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for d in docs:
        chunks.extend(chunk_document(d, chunk_size=chunk_size, overlap=overlap))
    return chunks
