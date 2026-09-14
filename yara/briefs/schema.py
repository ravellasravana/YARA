"""The output shape. Every claim carries chunk ids, which is what makes
grounding mechanically checkable instead of a matter of opinion."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class Citation(BaseModel):
    chunk_id: str
    source: str = ""
    snippet: str = ""

    @field_validator("snippet")
    @classmethod
    def _trim(cls, v: str) -> str:
        return v[:400]


class Finding(BaseModel):
    claim: str
    citations: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    subquestion_id: str | None = None

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    @property
    def is_grounded(self) -> bool:
        return bool(self.citations)


class Section(BaseModel):
    heading: str
    body: str
    citations: list[str] = Field(default_factory=list)


class SubQuestion(BaseModel):
    id: str
    question: str
    angle: str = "general"
    search_query: str = ""

    @model_validator(mode="after")
    def _default_query(self) -> "SubQuestion":
        # field_validator doesn't fire on an omitted field, so the default
        # only applied when someone passed "" explicitly. Caught by a test.
        if not self.search_query:
            self.search_query = self.question
        return self


class CritiqueIssue(BaseModel):
    claim: str
    issue: str
    severity: str = "medium"


class Critique(BaseModel):
    verdict: str = "pass"
    coverage: float = 0.0
    issues: list[CritiqueIssue] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"


class RunStats(BaseModel):
    subquestions: int = 0
    chunks_retrieved: int = 0
    findings: int = 0
    revisions: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: float = 0.0
    provider: str = ""
    degraded: bool = False


class ResearchBrief(BaseModel):
    run_id: str
    question: str
    summary: str = ""
    subquestions: list[SubQuestion] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    critique: Critique = Field(default_factory=Critique)
    stats: RunStats = Field(default_factory=RunStats)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def citation_coverage(self) -> float:
        if not self.findings:
            return 0.0
        return sum(1 for f in self.findings if f.is_grounded) / len(self.findings)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    def to_markdown(self) -> str:
        return render_markdown(self)


def render_markdown(brief: ResearchBrief) -> str:
    """Footnoted markdown.

    Only sources the body actually references get numbered. First version
    numbered off the whole citation list, but synthesis groups findings into
    sections and drops some, so the Sources list had entries nothing pointed
    at. Spotted it reading real output, not from a test.
    """
    # Determine what the body will reference before numbering anything.
    if brief.sections:
        referenced = {c for s in brief.sections for c in s.citations}
    else:
        referenced = {c for f in brief.findings for c in f.citations}

    used = [c for c in brief.citations if c.chunk_id in referenced]
    labels = {c.chunk_id: i + 1 for i, c in enumerate(used)}

    def marks(ids: list[str]) -> str:
        nums = sorted({labels[i] for i in ids if i in labels})
        return "".join(f"[^{n}]" for n in nums)

    lines = [f"# {brief.question}", ""]
    if brief.summary:
        lines += ["## Summary", "", brief.summary, ""]

    if brief.sections:
        lines += ["## Findings", ""]
        for section in brief.sections:
            lines += [f"### {section.heading}", "", f"{section.body}{marks(section.citations)}", ""]
    elif brief.findings:
        lines += ["## Findings", ""]
        for finding in brief.findings:
            lines.append(f"- {finding.claim}{marks(finding.citations)}")
        lines.append("")

    if brief.subquestions:
        lines += ["## Questions investigated", ""]
        lines += [f"{i + 1}. {sq.question} _({sq.angle})_" for i, sq in enumerate(brief.subquestions)]
        lines.append("")

    if brief.open_questions:
        lines += ["## Open questions", ""]
        lines += [f"- {q}" for q in brief.open_questions]
        lines.append("")

    lines += [
        "## Verification",
        "",
        f"- Verdict: **{brief.critique.verdict}**",
        f"- Citation coverage: {brief.citation_coverage:.0%}",
        f"- Revisions: {brief.stats.revisions}",
    ]
    if brief.critique.issues:
        lines.append("- Outstanding issues:")
        lines += [
            f"  - _{issue.severity}_ {issue.issue}: {issue.claim[:120]}"
            for issue in brief.critique.issues
        ]
    lines.append("")

    if used:
        lines += ["## Sources", ""]
        for citation in used:
            snippet = citation.snippet.replace("\n", " ")[:220]
            lines.append(
                f"[^{labels[citation.chunk_id]}]: `{citation.chunk_id}` "
                f"{citation.source} - {snippet}"
            )
        lines.append("")

    return "\n".join(lines)
