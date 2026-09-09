import pytest

from yara.memory.store import MemoryStore
from yara.retrieval.embeddings import HashingEmbedder


@pytest.fixture
def embedder():
    return HashingEmbedder(dim=256)


@pytest.fixture
def memory(embedder):
    store = MemoryStore(":memory:", embedder=embedder)
    yield store
    store.close()


class TestEpisodic:
    def test_run_lifecycle(self, memory):
        run_id = memory.start_run("why do agents fail?")
        record = memory.get_run(run_id)
        assert record.status == "running" and record.question == "why do agents fail?"

        memory.finish_run(run_id, status="completed", result={"summary": "done"})
        record = memory.get_run(run_id)
        assert record.status == "completed" and record.result["summary"] == "done"

    def test_steps_keep_insertion_order(self, memory):
        run_id = memory.start_run("q")
        for agent in ("planner", "analyst", "critic"):
            memory.log_step(run_id, agent=agent, action="complete")
        assert [s["agent"] for s in memory.get_steps(run_id)] == [
            "planner", "analyst", "critic"
        ]
        assert [s["ordinal"] for s in memory.get_steps(run_id)] == [0, 1, 2]

    def test_failure_is_recorded(self, memory):
        run_id = memory.start_run("q")
        memory.finish_run(run_id, status="failed", error="boom")
        assert memory.get_run(run_id).error == "boom"

    def test_unknown_run_returns_none(self, memory):
        assert memory.get_run("run_nope") is None


class TestScratchpad:
    def test_set_get_and_overwrite(self, memory):
        run_id = memory.start_run("q")
        memory.scratch_set(run_id, "findings", [{"claim": "x"}])
        assert memory.scratch_get(run_id, "findings") == [{"claim": "x"}]
        memory.scratch_set(run_id, "findings", [])
        assert memory.scratch_get(run_id, "findings") == []

    def test_scoped_per_run(self, memory):
        a, b = memory.start_run("a"), memory.start_run("b")
        memory.scratch_set(a, "k", 1)
        assert memory.scratch_get(b, "k") is None

    def test_missing_key_returns_default(self, memory):
        assert memory.scratch_get(memory.start_run("q"), "absent", "fallback") == "fallback"


class TestSemantic:
    def test_recall_ranks_by_similarity(self, memory):
        memory.remember("Hybrid retrieval fuses dense and lexical rankings.", confidence=0.9)
        memory.remember("Sourdough needs a mature starter.", confidence=0.9)
        facts = memory.recall("dense and lexical retrieval fusion", k=2)
        assert facts and "Hybrid retrieval" in facts[0].text

    def test_recall_filters_by_kind(self, memory):
        memory.remember("A finding.", kind="finding")
        memory.remember("A definition.", kind="definition")
        assert all(f.kind == "definition" for f in memory.recall("definition", kind="definition"))

    def test_recall_on_empty_store(self, memory):
        assert memory.recall("anything") == []

    def test_facts_survive_reopen(self, tmp_path, embedder):
        path = tmp_path / "m.sqlite3"
        with MemoryStore(path, embedder=embedder) as store:
            store.remember("Reciprocal rank fusion needs no score calibration.")
        with MemoryStore(path, embedder=embedder) as store:
            assert store.recall("rank fusion calibration", k=1)

    def test_remember_is_idempotent(self, memory):
        """Repeated runs rediscover the same claim; memory must not grow per run."""
        first = memory.remember("Recall at k bounds answer quality.", citations=["a"])
        second = memory.remember("Recall at k bounds answer quality.", citations=["b"])
        assert first == second
        assert memory.count("facts") == 1
        fact = memory.recall("recall at k answer quality", k=1)[0]
        assert fact.citations == ["a", "b"]

    def test_same_text_under_a_different_kind_is_separate(self, memory):
        memory.remember("Shared text.", kind="finding")
        memory.remember("Shared text.", kind="definition")
        assert memory.count("facts") == 2

    def test_count_rejects_unknown_table(self, memory):
        with pytest.raises(ValueError):
            memory.count("; DROP TABLE runs")
