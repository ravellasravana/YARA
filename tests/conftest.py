import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yara.config import Settings
from yara.memory.store import MemoryStore
from yara.retrieval.chunking import Document, chunk_document
from yara.retrieval.embeddings import HashingEmbedder
from yara.retrieval.retriever import HybridRetriever
from yara.retrieval.vector_store import VectorStore

CORPUS = Path(__file__).resolve().parents[1] / "examples" / "corpus"


@pytest.fixture
def embedder():
    return HashingEmbedder(dim=256)


@pytest.fixture
def settings(tmp_path):
    return Settings(
        provider="echo", embedder="hashing", embedding_dim=256,
        data_dir=tmp_path / "data", max_subquestions=3, top_k=5,
    )


@pytest.fixture
def memory(embedder):
    store = MemoryStore(":memory:", embedder=embedder)
    yield store
    store.close()


@pytest.fixture
def sample_doc():
    return Document.from_text(
        "# Alpha\n"
        "Retrieval augmented generation grounds output in retrieved passages. "
        "Hallucination rates fall when grounding is enforced strictly.\n"
        "# Beta\n"
        "BM25 matches rare literal tokens such as identifier XK9427 reliably. "
        "Dense retrieval generalises across paraphrase instead.",
        title="Notes", source="notes.md",
    )


@pytest.fixture
def retriever(embedder, sample_doc):
    store = VectorStore(embedder)
    r = HybridRetriever(store)
    r.add(chunk_document(sample_doc, chunk_size=200, overlap=40))
    return r


@pytest.fixture
def yara(settings):
    from yara.app import YARA

    instance = YARA(settings, persist=False)
    instance.ingest_path(CORPUS)
    yield instance
    instance.close()
