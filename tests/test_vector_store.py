import pytest

from yara.retrieval.chunking import Document, chunk_document
from yara.retrieval.embeddings import HashingEmbedder
from yara.retrieval.vector_store import VectorStore


@pytest.fixture
def embedder():
    return HashingEmbedder(dim=256)


@pytest.fixture
def chunks():
    doc = Document.from_text(
        "# Alpha\n"
        "Retrieval augmented generation grounds output in retrieved passages. "
        "Hallucination rates fall when grounding is enforced strictly.\n"
        "# Beta\n"
        "BM25 matches rare literal tokens such as identifier XK9427 reliably. "
        "Dense retrieval generalises across paraphrase instead.",
        title="Notes",
        source="notes.md",
    )
    return chunk_document(doc, chunk_size=200, overlap=40)


def test_add_and_search(embedder, chunks):
    store = VectorStore(embedder)
    assert store.add(chunks) > 0
    hits = store.search("hallucination grounding", k=2)
    assert hits
    assert hits[0][1] > 0


def test_add_is_idempotent(embedder, chunks):
    store = VectorStore(embedder)
    first = store.add(chunks)
    assert store.add(chunks) == 0
    assert len(store) == first


def test_results_come_back_sorted(embedder, chunks):
    store = VectorStore(embedder)
    store.add(chunks)
    scores = [s for _, s in store.search("retrieval passages", k=5)]
    assert scores == sorted(scores, reverse=True)


def test_get_by_id(embedder, chunks):
    store = VectorStore(embedder)
    store.add(chunks)
    assert store.get(chunks[0].chunk_id).chunk_id == chunks[0].chunk_id
    assert store.get("nope") is None


def test_roundtrip_persistence(embedder, chunks, tmp_path):
    store = VectorStore(embedder)
    store.add(chunks)
    store.save(tmp_path / "idx")

    reloaded = VectorStore.load(tmp_path / "idx", embedder)
    assert len(reloaded) == len(store)
    assert reloaded.search("BM25 identifier", k=1)[0][0].chunk_id


def test_load_discards_index_from_a_different_embedder(embedder, chunks, tmp_path):
    """Vectors from another embedder aren't comparable - drop them, don't compare."""
    store = VectorStore(embedder)
    store.add(chunks)
    store.save(tmp_path / "idx")
    assert len(VectorStore.load(tmp_path / "idx", HashingEmbedder(dim=64))) == 0


def test_load_missing_index_starts_empty(embedder, tmp_path):
    assert len(VectorStore.load(tmp_path / "nothing-here", embedder)) == 0


def test_empty_store_search_is_safe(embedder):
    assert VectorStore(embedder).search("anything", k=5) == []
