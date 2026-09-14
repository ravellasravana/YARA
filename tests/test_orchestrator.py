"""Pipeline tests against the deterministic provider. A failure here means
the orchestration changed, not that a model drifted."""

import pytest

from yara.briefs.schema import Finding
from yara.orchestrator.orchestrator import RunState, _dedupe


class TestDedupe:
    def test_merges_identical_claims_and_unions_citations(self):
        merged = _dedupe([
            Finding(claim="Same claim.", citations=["a"], confidence=0.4),
            Finding(claim="  same   CLAIM. ", citations=["b"], confidence=0.8),
        ])
        assert len(merged) == 1
        assert merged[0].citations == ["a", "b"]
        assert merged[0].confidence == 0.8

    def test_keeps_distinct_claims(self):
        assert len(_dedupe([Finding(claim="A"), Finding(claim="B")])) == 2


class TestPipeline:
    def test_produces_a_grounded_brief(self, yara):
        brief = yara.research("How does hybrid retrieval reduce hallucination?")
        assert brief.summary
        assert brief.findings
        assert brief.citations
        assert brief.citation_coverage == 1.0

    def test_every_citation_resolves_to_retrieved_evidence(self, yara):
        """The invariant the whole thing is built around."""
        brief = yara.research("What bounds the quality of a RAG answer?")
        valid = {c.chunk_id for c in brief.citations}
        for f in brief.findings:
            assert set(f.citations) <= valid
        for s in brief.sections:
            assert set(s.citations) <= valid

    def test_unresolvable_citations_are_stripped_in_code(self, yara):
        """A fabricated chunk id is gone before the critic ever sees it."""
        run_id = yara.memory.start_run("q")
        state = RunState(run_id=run_id, question="q")
        real = yara.retriever.retrieve("retrieval", k=1)[0]
        state.evidence = {real.chunk_id: real}
        state.findings = [
            Finding(claim="grounded", citations=[real.chunk_id]),
            Finding(claim="fabricated", citations=["chunk_that_never_existed"]),
            Finding(claim="uncited", citations=[]),
        ]
        yara.orchestrator._verify_citations(state)
        assert [f.claim for f in state.findings] == ["grounded"]

    def test_run_is_traceable_end_to_end(self, yara):
        brief = yara.research("How is grounded generation evaluated?")
        agents = {s["agent"] for s in yara.memory.get_steps(brief.run_id)}
        assert {"planner", "retriever", "analyst", "verifier", "synthesiser"} <= agents

    def test_brief_is_retrievable_after_the_run(self, yara):
        brief = yara.research("What is reciprocal rank fusion?")
        assert yara.get_brief(brief.run_id).run_id == brief.run_id

    def test_empty_question_rejected(self, yara):
        with pytest.raises(ValueError):
            yara.research("   ")

    def test_empty_corpus_degrades_without_crashing(self, settings):
        from yara.app import YARA

        with YARA(settings, persist=False) as bare:
            brief = bare.research("anything at all")
            assert brief.findings == []
            assert not brief.critique.passed

    def test_memory_carries_across_runs(self, yara):
        first = yara.research("How does hybrid retrieval reduce hallucination?")
        assert first.critique.passed
        assert yara.memory.count("facts") > 0
        assert yara.memory.recall("hybrid retrieval hallucination", k=1)

    def test_stats_are_populated(self, yara):
        stats = yara.research("What is chunking?").stats
        assert stats.subquestions > 0 and stats.chunks_retrieved > 0
        assert stats.output_tokens > 0 and stats.provider == "echo"

    def test_every_footnote_is_referenced_in_the_body(self, yara):
        import re

        md = yara.research("How does hybrid retrieval reduce hallucination?").to_markdown()
        body, _, sources = md.partition("## Sources")
        referenced = set(re.findall(r"\[\^(\d+)\]", body))
        listed = set(re.findall(r"^\[\^(\d+)\]:", sources, re.MULTILINE))
        assert listed == referenced
        assert listed
