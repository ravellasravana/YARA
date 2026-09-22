"""The HTTP layer, against an in-memory facade with the example corpus."""

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from yara.api import create_app  # noqa: E402


@pytest.fixture
def client(yara):
    # `with` runs the lifespan hook, same as a real server start.
    with TestClient(create_app(yara)) as c:
        yield c


@pytest.fixture
def empty_client(settings):
    from yara.app import YARA

    y = YARA(settings, persist=False)
    with TestClient(create_app(y)) as c:
        yield c
    y.close()


def test_health(client, yara):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["provider"] == "echo"
    assert body["corpus_size"] == yara.corpus_size > 0


def test_ingest_text_documents(empty_client):
    payload = {"documents": [{"text": "BM25 ranks by term frequency.", "title": "bm25"}]}
    r = empty_client.post("/ingest", json=payload)
    assert r.status_code == 200
    assert r.json() == {"added": 1, "corpus_size": 1}
    # Same document again: chunk ids hash source, title and text, so a retry
    # (say, after a client timeout) can't duplicate the corpus.
    again = empty_client.post("/ingest", json=payload)
    assert again.json() == {"added": 0, "corpus_size": 1}


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"documents": []},
        {"documents": [{"text": ""}]},
        {"path": "/etc/passwd"},  # server-side paths are not accepted
    ],
)
def test_ingest_validation(empty_client, payload):
    assert empty_client.post("/ingest", json=payload).status_code == 422


def test_research_returns_a_brief_with_verified_citations(client):
    r = client.post("/research", json={"question": "How does retrieval reduce hallucination?"})
    assert r.status_code == 200
    brief = r.json()
    assert brief["run_id"].startswith("run_")
    assert brief["findings"]
    known = {c["chunk_id"] for c in brief["citations"]}
    cited = {cid for f in brief["findings"] for cid in f["citations"]}
    assert cited and cited <= known


def test_research_on_empty_corpus_is_a_conflict(empty_client):
    r = empty_client.post("/research", json={"question": "anything"})
    assert r.status_code == 409
    assert "ingest" in r.json()["detail"]


@pytest.mark.parametrize("question", ["", "   "])
def test_research_rejects_blank_questions(client, question):
    assert client.post("/research", json={"question": question}).status_code == 422


def test_fetch_a_past_brief_as_json_and_markdown(client):
    run_id = client.post("/research", json={"question": "What is BM25?"}).json()["run_id"]

    r = client.get(f"/research/{run_id}")
    assert r.status_code == 200
    assert r.json()["question"] == "What is BM25?"

    md = client.get(f"/research/{run_id}/markdown")
    assert md.status_code == 200
    assert md.headers["content-type"].startswith("text/markdown")
    assert md.text.startswith("# What is BM25?")


def test_unknown_run_is_404(client):
    assert client.get("/research/run_doesnotexist").status_code == 404
    assert client.get("/research/run_doesnotexist/markdown").status_code == 404


def test_decide(client):
    r = client.post(
        "/decide",
        json={
            "data": [{"name": "a", "price": 10}, {"name": "b", "price": 5}],
            "criteria": {"price": -1},
        },
    )
    assert r.status_code == 200
    assert r.json()["recommendations"][0]["option"]["name"] == "b"


def test_decide_with_no_usable_options_is_422(client):
    r = client.post("/decide", json={"data": []})
    assert r.status_code == 422
    assert "No options" in r.json()["detail"]


def test_app_builds_and_closes_its_own_facade(settings):
    """With no facade passed in, startup builds one from settings."""
    app = create_app(settings=settings)
    with TestClient(app) as c:
        assert c.get("/health").json()["corpus_size"] == 0
    assert (settings.data_dir / "yara.sqlite3").exists()
