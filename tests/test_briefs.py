from yara.briefs.schema import (
    Citation,
    Critique,
    Finding,
    ResearchBrief,
    Section,
    SubQuestion,
)


def test_confidence_is_clamped():
    assert Finding(claim="x", confidence=9).confidence == 1.0
    assert Finding(claim="x", confidence=-1).confidence == 0.0


def test_subquestion_search_query_defaults_to_question():
    assert SubQuestion(id="sq1", question="What is X?").search_query == "What is X?"


def test_critique_passed():
    assert Critique(verdict="pass").passed
    assert not Critique(verdict="revise").passed


def test_citation_coverage():
    brief = ResearchBrief(
        run_id="r", question="q",
        findings=[Finding(claim="a", citations=["c1"]), Finding(claim="b")],
    )
    assert brief.citation_coverage == 0.5


def test_markdown_only_footnotes_referenced_sources():
    """Sections cite c1. c2 is on the brief but unreferenced - it must not
    show up in Sources, otherwise there's a footnote nothing points at."""
    brief = ResearchBrief(
        run_id="r", question="q", summary="s",
        sections=[Section(heading="H", body="Body.", citations=["c1"])],
        citations=[
            Citation(chunk_id="c1", source="a", snippet="A"),
            Citation(chunk_id="c2", source="b", snippet="B"),
        ],
    )
    md = brief.to_markdown()
    assert "[^1]" in md
    assert "[^2]" not in md
    assert "`c2`" not in md


def test_markdown_falls_back_to_findings_when_no_sections():
    brief = ResearchBrief(
        run_id="r", question="q",
        findings=[Finding(claim="A claim.", citations=["c1"])],
        citations=[Citation(chunk_id="c1", source="a", snippet="A")],
    )
    md = brief.to_markdown()
    assert "- A claim.[^1]" in md


def test_roundtrips_through_dict():
    brief = ResearchBrief(run_id="r", question="q", findings=[Finding(claim="x")])
    assert ResearchBrief.model_validate(brief.to_dict()).findings[0].claim == "x"
