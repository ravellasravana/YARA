import pytest

from yara.retrieval.chunking import Document, chunk_document


@pytest.fixture
def doc():
    return Document.from_text(
        "# Alpha\n"
        "Retrieval augmented generation grounds output in retrieved passages. "
        "Hallucination rates fall when grounding is enforced strictly.\n"
        "# Beta\n"
        "BM25 matches rare literal tokens such as identifier XK9427 reliably. "
        "Dense retrieval generalises across paraphrase instead.",
        title="Notes",
        source="notes.md",
    )


def test_respects_size_budget(doc):
    chunks = chunk_document(doc, chunk_size=120, overlap=20)
    assert chunks
    # One sentence of slack - packing never splits mid-sentence.
    assert all(len(c.text) <= 240 for c in chunks)


def test_chunk_ids_unique_and_ordered(doc):
    chunks = chunk_document(doc, chunk_size=150, overlap=30)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_headings_become_sections(doc):
    sections = {c.section for c in chunk_document(doc, chunk_size=400, overlap=0)}
    assert {"Alpha", "Beta"} <= sections


def test_oversized_sentence_is_split_not_dropped():
    doc = Document.from_text("x" * 500 + ".", title="t")
    chunks = chunk_document(doc, chunk_size=100, overlap=10)
    assert sum(len(c.text) for c in chunks) >= 500


def test_rejects_overlap_larger_than_chunk(doc):
    with pytest.raises(ValueError):
        chunk_document(doc, chunk_size=100, overlap=100)


def test_citation_label_includes_section(doc):
    chunk = chunk_document(doc, chunk_size=400, overlap=0)[0]
    assert "Notes" in chunk.citation_label()
