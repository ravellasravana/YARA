"""Dense vector index. Exact cosine search, persisted as npy + jsonl."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .chunking import Chunk
from .embeddings import Embedder, l2_normalize

logger = logging.getLogger("yara.retrieval")


class VectorStore:
    """Brute-force cosine search over a float32 matrix.

    Exact rather than approximate on purpose - at the corpus sizes this is
    aimed at, recall matters more than latency, and an exact index has no
    tuning knobs to get wrong. FaissVectorStore below swaps in ANN with the
    same interface when that stops being true.
    """

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.dim = embedder.dim
        self._vectors = np.zeros((0, self.dim), dtype=np.float32)
        self._chunks: list[Chunk] = []
        self._seen: set[str] = set()

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def add(self, chunks: list[Chunk]) -> int:
        """Returns how many were actually added. Ids already present are skipped."""
        fresh = [c for c in chunks if c.chunk_id not in self._seen]
        if not fresh:
            return 0

        vecs = self.embedder.encode([c.text for c in fresh])
        if vecs.shape[1] != self.dim:
            raise ValueError(f"embedder gave dim {vecs.shape[1]}, index wants {self.dim}")

        self._vectors = np.vstack([self._vectors, vecs])
        self._chunks.extend(fresh)
        self._seen.update(c.chunk_id for c in fresh)
        return len(fresh)

    def search(self, query: str, k: int = 8) -> list[tuple[Chunk, float]]:
        if not self._chunks or k <= 0:
            return []

        q = l2_normalize(self.embedder.encode([query]))[0]
        scores = self._vectors @ q
        k = min(k, len(scores))

        # argpartition gets the top k without sorting everything, then sort those.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self._chunks[i], float(scores[i])) for i in top]

    def get(self, chunk_id: str) -> Chunk | None:
        for c in self._chunks:
            if c.chunk_id == chunk_id:
                return c
        return None

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        np.save(path / "vectors.npy", self._vectors)
        with (path / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self._chunks:
                f.write(json.dumps(c.to_dict()) + "\n")
        (path / "meta.json").write_text(
            json.dumps({"dim": self.dim, "embedder": self.embedder.name, "count": len(self)})
        )
        logger.info("saved index of %d chunks to %s", len(self), path)

    @classmethod
    def load(cls, path: Path, embedder: Embedder) -> "VectorStore":
        path = Path(path)
        store = cls(embedder)

        vectors_file = path / "vectors.npy"
        chunks_file = path / "chunks.jsonl"
        if not vectors_file.exists() or not chunks_file.exists():
            logger.info("no index at %s, starting empty", path)
            return store

        # Vectors from a different embedder aren't comparable. Better to throw
        # the index away and force a re-ingest than to silently return garbage.
        meta_file = path / "meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            if meta.get("embedder") != embedder.name or meta.get("dim") != embedder.dim:
                logger.warning(
                    "index at %s built with %s/dim=%s, embedder is %s/dim=%s - discarding, re-ingest needed",
                    path, meta.get("embedder"), meta.get("dim"), embedder.name, embedder.dim,
                )
                return store

        store._vectors = np.load(vectors_file).astype(np.float32)
        with chunks_file.open(encoding="utf-8") as f:
            store._chunks = [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]
        store._seen = {c.chunk_id for c in store._chunks}

        if len(store._chunks) != store._vectors.shape[0]:
            raise ValueError(
                f"corrupt index at {path}: {len(store._chunks)} chunks, "
                f"{store._vectors.shape[0]} vectors"
            )

        logger.info("loaded index of %d chunks from %s", len(store), path)
        return store


class FaissVectorStore(VectorStore):
    """ANN version for when the corpus outgrows brute force. Not exercised much."""

    def __init__(self, embedder: Embedder, nlist: int = 100):
        super().__init__(embedder)
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("faiss-cpu not installed. pip install faiss-cpu") from exc
        self._index = faiss.IndexFlatIP(self.dim)
        self.nlist = nlist

    def add(self, chunks: list[Chunk]) -> int:
        before = len(self._chunks)
        added = super().add(chunks)
        if added:
            self._index.add(self._vectors[before:])
        return added

    def search(self, query: str, k: int = 8) -> list[tuple[Chunk, float]]:
        if not self._chunks or k <= 0:
            return []
        q = l2_normalize(self.embedder.encode([query]))
        scores, idx = self._index.search(q, min(k, len(self._chunks)))
        return [
            (self._chunks[i], float(s))
            for i, s in zip(idx[0], scores[0], strict=False)
            if i != -1
        ]
