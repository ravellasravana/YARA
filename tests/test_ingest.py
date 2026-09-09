import pytest

from yara.retrieval.ingest import load_path, load_text_file, strip_html, to_chunks


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "a.md").write_text("# Topic\nFirst document body here.", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Second document, plain text.", encoding="utf-8")
    (tmp_path / "skip.bin").write_bytes(b"\x00\x01\x02")
    sub = tmp_path / "nested"
    sub.mkdir()
    (sub / "c.md").write_text("# Nested\nThird document.", encoding="utf-8")
    return tmp_path


def test_loads_a_single_file(corpus):
    doc = load_text_file(corpus / "a.md")
    assert doc.title == "a"
    assert "First document" in doc.text


def test_walks_a_directory_recursively(corpus):
    docs = load_path(corpus)
    assert {d.title for d in docs} == {"a", "b", "c"}


def test_skips_unsupported_extensions(corpus):
    assert all(not d.source.endswith(".bin") for d in load_path(corpus))


def test_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_path(tmp_path / "not-here")


def test_unreadable_file_does_not_abort_the_directory(corpus, monkeypatch):
    """One bad file shouldn't cost you the whole ingest."""
    import yara.retrieval.ingest as ingest

    real = ingest._load_one

    def explode(path):
        if path.name == "b.txt":
            raise OSError("boom")
        return real(path)

    monkeypatch.setattr(ingest, "_load_one", explode)
    assert {d.title for d in load_path(corpus)} == {"a", "c"}


def test_to_chunks_flattens_documents(corpus):
    chunks = to_chunks(load_path(corpus), chunk_size=200, overlap=20)
    assert len(chunks) >= 3
    assert len({c.chunk_id for c in chunks}) == len(chunks)


class TestStripHtml:
    def test_drops_tags_and_keeps_text(self):
        assert "Hello" in strip_html("<p>Hello</p>")

    def test_drops_script_and_style_bodies(self):
        out = strip_html("<script>var secret=1</script><p>Visible</p><style>a{}</style>")
        assert "Visible" in out
        assert "secret" not in out

    def test_unescapes_common_entities(self):
        assert "&" in strip_html("<p>Tom &amp; Jerry</p>")

    def test_html_file_is_stripped_on_load(self, tmp_path):
        f = tmp_path / "page.html"
        f.write_text("<html><body><p>Body text</p></body></html>", encoding="utf-8")
        doc = load_text_file(f)
        assert "Body text" in doc.text
        assert "<p>" not in doc.text
