"""Text to vectors. Default backend needs no model download and no network."""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger("yara.retrieval")

TOKEN = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Embedder(Protocol):
    dim: int
    name: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class HashingEmbedder:
    """Feature hashing over unigrams and bigrams.

    Weights are non-negative on purpose. The first version of this used signed
    hashing, which is the standard trick and keeps inner products unbiased -
    but two colliding terms with opposite signs cancel, and a passage holding
    both query terms scored exactly 0.0. Took a while to find. Unsigned turns
    collisions into noise instead, which just degrades ranking.

    Unigrams and bigrams get separate halves of the vector so they can't
    collide with each other.
    """

    name = "hashing"

    def __init__(self, dim: int = 1024, use_bigrams: bool = True):
        if dim <= 0:
            raise ValueError("dim must be positive")
        if use_bigrams and dim < 2:
            raise ValueError("dim must be at least 2 with bigrams enabled")
        self.dim = dim
        self.use_bigrams = use_bigrams
        self.split = dim // 2 if use_bigrams else dim

    def encode(self, texts: list[str]) -> np.ndarray:
        m = np.zeros((len(texts), self.dim), dtype=np.float32)

        for row, text in enumerate(texts):
            words = tokenize(text)

            for term, count in Counter(words).items():
                m[row, self._bucket(term, 0, self.split)] += 1 + math.log(count)

            if self.use_bigrams:
                pairs = [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]
                for term, count in Counter(pairs).items():
                    # Bigrams corroborate, they don't lead. Half weight.
                    m[row, self._bucket(term, self.split, self.dim)] += 0.5 * (1 + math.log(count))

        return l2_normalize(m)

    def _bucket(self, term: str, lo: int, hi: int) -> int:
        h = hashlib.blake2b(term.encode(), digest_size=8).digest()
        return lo + int.from_bytes(h, "big") % (hi - lo)


class SentenceTransformerEmbedder:
    name = "sentence-transformers"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers not installed. pip install yara-research[neural], "
                "or set YARA_EMBEDDER=hashing."
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        v = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return l2_normalize(np.asarray(v, dtype=np.float32))


def get_embedder(name: str = "hashing", *, dim: int = 1024) -> Embedder:
    if name == "hashing":
        return HashingEmbedder(dim)
    if name == "sentence-transformers":
        return SentenceTransformerEmbedder()
    raise ValueError(f"unknown embedder {name!r}")
