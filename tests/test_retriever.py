import pytest

from yara.retrieval.chunking import Document, chunk_document
from yara.retrieval.embeddings import HashingEmbedder
from yara.retrieval.retriever import BM25Index, HybridRetriever
from yara.retrieval.vector_store import VectorStore


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


@pytest.fixture
def retriever(chunks):
    r = HybridRetriever(VectorStore(HashingEmbedder(dim=256)))
    r.add(chunks)
    return r


class TestBM25:
    def test_finds_rare_literal_token(self, chunks):
        idx = BM25Index()
        idx.add(chunks)
        hits = idx.search("XK9427", k=1)
        assert hits
        assert "XK9427" in hits[0][0].text

    def test_empty_index_is_safe(self):
        assert BM25Index().search("anything") == []

    def test_unmatched_query_returns_nothing(self, chunks):
        idx = BM25Index()
        idx.add(chunks)
        assert idx.search("zzzqqqxyw") == []


class TestFusion:
    def test_results_are_ranked(self, retriever):
        out = retriever.retrieve("hallucination grounding", k=3)
        assert out
        assert out == sorted(out, key=lambda r: (-r.score, r.chunk_id))

    def test_surfaces_lexical_only_match(self, retriever):
        """The whole point of the hybrid. A hashing embedder can't see a rare
        identifier; BM25 pulls it back into the top results."""
        out = retriever.retrieve("XK9427", k=3)
        assert any("XK9427" in r.chunk.text for r in out)

    def test_records_which_side_matched(self, retriever):
        out = retriever.retrieve("grounding", k=3)
        assert any(r.dense_rank is not None or r.lexical_rank is not None for r in out)

    def test_multi_query_deduplicates(self, retriever):
        out = retriever.retrieve_multi(
            ["hallucination grounding", "grounding hallucination rates"], k=5
        )
        assert len({r.chunk_id for r in out}) == len(out)

    def test_empty_query_returns_nothing(self, retriever):
        assert retriever.retrieve("   ", k=3) == []

    def test_add_is_idempotent(self, retriever, chunks):
        assert retriever.add(chunks) == 0

    def test_bm25_backfilled_from_a_prepopulated_store(self, chunks):
        """Loading an index off disk gives a full store and an empty BM25."""
        store = VectorStore(HashingEmbedder(dim=256))
        store.add(chunks)
        r = HybridRetriever(store)
        assert len(r.bm25) == len(chunks)
