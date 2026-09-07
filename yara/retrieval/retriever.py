"""Hybrid retrieval - dense vectors fused with BM25."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .chunking import Chunk
from .embeddings import tokenize
from .vector_store import VectorStore


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


class BM25Index:
    """Okapi BM25 over the same chunks the vector store holds.

    Written out rather than pulled in as a dependency - it's forty lines and
    the constants matter enough to want them visible.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._tf: list[dict[str, int]] = []
        self._df: dict[str, int] = defaultdict(int)
        self._lengths: list[int] = []
        self._avg_length = 0.0

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: Iterable[Chunk]) -> None:
        for chunk in chunks:
            tokens = tokenize(chunk.text)
            freqs: dict[str, int] = defaultdict(int)
            for t in tokens:
                freqs[t] += 1

            self._chunks.append(chunk)
            self._tf.append(dict(freqs))
            self._lengths.append(len(tokens))
            for term in freqs:
                self._df[term] += 1

        if self._lengths:
            self._avg_length = sum(self._lengths) / len(self._lengths)

    def search(self, query: str, k: int = 8) -> list[tuple[Chunk, float]]:
        if not self._chunks:
            return []

        n = len(self._chunks)
        scores = [0.0] * n

        for term in tokenize(query):
            df = self._df.get(term)
            if not df:
                continue

            # +1 inside the log keeps this non-negative for terms in most docs.
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))

            for i, freqs in enumerate(self._tf):
                tf = freqs.get(term)
                if not tf:
                    continue
                norm = 1 - self.b + self.b * (self._lengths[i] / (self._avg_length or 1))
                scores[i] += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * norm)

        ranked = sorted(
            ((self._chunks[i], s) for i, s in enumerate(scores) if s > 0),
            key=lambda pair: -pair[1],
        )
        return ranked[:k]


class HybridRetriever:
    """Fuses dense and lexical results with reciprocal rank fusion.

    Dense search generalises across paraphrase but goes blind on rare literal
    tokens - identifiers, acronyms, version numbers. BM25 is the other way
    round. RRF combines the two ranked lists on rank rather than score, so the
    two scales never need calibrating against each other. That's why it moves
    to a new corpus without retuning.
    """

    def __init__(self, vector_store: VectorStore, bm25: BM25Index | None = None, rrf_k: int = 60):
        self.vector_store = vector_store
        self.bm25 = bm25 if bm25 is not None else BM25Index()
        self.rrf_k = rrf_k

        # Handles being handed an already-populated store (e.g. loaded from disk).
        if len(self.bm25) == 0 and len(vector_store) > 0:
            self.bm25.add(vector_store.chunks)

    def add(self, chunks: list[Chunk]) -> int:
        added = self.vector_store.add(chunks)
        if added:
            existing = {c.chunk_id for c in self.vector_store.chunks[:-added]}
            self.bm25.add([c for c in chunks if c.chunk_id not in existing])
        return added

    def retrieve(self, query: str, k: int = 8, pool: int | None = None) -> list[ScoredChunk]:
        if not query or not query.strip():
            return []

        # Pull deeper than k from each side so fusion has something to work with.
        pool = pool or max(k * 3, 20)
        dense = self.vector_store.search(query, k=pool)
        lexical = self.bm25.search(query, k=pool)

        dense_ranks = {c.chunk_id: i for i, (c, _) in enumerate(dense)}
        lex_ranks = {c.chunk_id: i for i, (c, _) in enumerate(lexical)}

        by_id: dict[str, Chunk] = {c.chunk_id: c for c, _ in dense}
        by_id.update({c.chunk_id: c for c, _ in lexical})

        fused = []
        for chunk_id, chunk in by_id.items():
            score = 0.0
            dr = dense_ranks.get(chunk_id)
            lr = lex_ranks.get(chunk_id)
            if dr is not None:
                score += 1.0 / (self.rrf_k + dr + 1)
            if lr is not None:
                score += 1.0 / (self.rrf_k + lr + 1)
            fused.append(ScoredChunk(chunk, score, dr, lr))

        # chunk_id as tiebreak so results are stable across runs.
        fused.sort(key=lambda sc: (-sc.score, sc.chunk_id))
        return fused[:k]

    def retrieve_multi(
        self, queries: list[str], k: int = 8, per_query: int | None = None
    ) -> list[ScoredChunk]:
        """Fuse across several queries, deduplicating. Chunks hit by more than
        one query accumulate score, which is usually what you want."""
        per_query = per_query or k
        best: dict[str, ScoredChunk] = {}

        for q in queries:
            for sc in self.retrieve(q, k=per_query):
                if sc.chunk_id in best:
                    best[sc.chunk_id].score += sc.score
                else:
                    best[sc.chunk_id] = sc

        return sorted(best.values(), key=lambda sc: (-sc.score, sc.chunk_id))[:k]
