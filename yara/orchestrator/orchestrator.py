"""The run. A bounded state machine, not an open-ended agent loop.

    recall -> plan -> [retrieve -> analyse]* -> verify -> synthesise
                                                              |
                                          revise (capped) <---+--> critique

Two decisions carry most of the weight here.

Citations are checked in code before the critic sees anything. Anything that
doesn't resolve to a chunk retrieved this run gets stripped. A model asked to
police its own citation ids approves the ones it made up - the distribution
that produced the fake id also finds it plausible. A set membership test
doesn't care.

The revision loop is capped and it fails open. If the critic still objects
after max_revisions, the brief comes back with the verdict and the issues
attached rather than looping or quietly claiming success.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ..agents.research import AnalysisAgent, CriticAgent, PlannerAgent, SynthesisAgent
from ..briefs.schema import (
    Citation,
    Critique,
    Finding,
    ResearchBrief,
    RunStats,
    Section,
    SubQuestion,
)
from ..config import Settings, get_settings
from ..llm.base import LLMProvider, Usage
from ..memory.store import MemoryStore
from ..retrieval.retriever import HybridRetriever, ScoredChunk
from ..tools.base import ToolRegistry

logger = logging.getLogger("yara.orchestrator")


@dataclass
class RunState:
    """Mutable state threaded through one run."""

    run_id: str
    question: str
    subquestions: list[SubQuestion] = field(default_factory=list)
    evidence: dict[str, ScoredChunk] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    draft: dict[str, Any] = field(default_factory=dict)
    critique: Critique = field(default_factory=Critique)
    revisions: int = 0
    tool_calls: int = 0
    usage: Usage = field(default_factory=Usage)
    degraded: bool = False

    @property
    def evidence_ids(self) -> list[str]:
        return list(self.evidence)


class Orchestrator:
    def __init__(
        self,
        provider: LLMProvider,
        retriever: HybridRetriever,
        memory: MemoryStore,
        *,
        settings: Settings | None = None,
        tools: ToolRegistry | None = None,
    ):
        self.settings = settings or get_settings()
        self.provider = provider
        self.retriever = retriever
        self.memory = memory
        self.tools = tools

        shared = {
            "tools": tools,
            "memory": memory,
            "max_iterations": self.settings.max_tool_iterations,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
        }
        self.planner = PlannerAgent(provider, **shared)
        self.analyst = AnalysisAgent(provider, **shared)
        self.synthesiser = SynthesisAgent(provider, **shared)
        self.critic = CriticAgent(provider, **shared)

    # ---------- public entry point ----------

    def run(self, question: str, *, use_tools: bool = False) -> ResearchBrief:
        if not question or not question.strip():
            raise ValueError("question must not be empty")

        started = time.perf_counter()
        run_id = self.memory.start_run(question, provider=self.provider.name)
        state = RunState(run_id=run_id, question=question.strip())

        try:
            self._plan(state)
            self._gather(state, use_tools=use_tools)
            self._verify_citations(state)
            self._synthesise_and_critique(state)
            brief = self._assemble(state, started)
            self._persist(state, brief)
            self.memory.finish_run(run_id, status="completed", result=brief.to_dict())
            return brief
        except Exception as exc:  # noqa: BLE001 - record the failure before surfacing
            logger.exception("run %s failed", run_id)
            self.memory.finish_run(run_id, status="failed", error=f"{type(exc).__name__}: {exc}")
            raise

    # ---------- stages ----------

    def _plan(self, state: RunState) -> None:
        prior = self.memory.recall(state.question, k=3)
        prior_text = "\n".join(f"- {f.text}" for f in prior)
        if prior_text:
            logger.info("run %s recalled %d prior facts", state.run_id, len(prior))

        result = self.planner.run(
            run_id=state.run_id,
            question=state.question,
            max_subquestions=self.settings.max_subquestions,
            prior=prior_text,
        )
        state.usage = state.usage + result.usage
        state.tool_calls += result.tool_calls
        state.degraded |= not result.ok

        subquestions = result.output or []
        if not subquestions:
            subquestions = [
                SubQuestion(
                    id="sq1", question=state.question, angle="general",
                    search_query=state.question,
                )
            ]
        state.subquestions = subquestions[: self.settings.max_subquestions]
        self.memory.scratch_set(
            state.run_id, "subquestions", [sq.model_dump() for sq in state.subquestions]
        )

    def _gather(self, state: RunState, *, use_tools: bool) -> None:
        """Retrieve and analyse per sub-question, accumulating findings."""
        for subquestion in state.subquestions:
            hits = self.retriever.retrieve(
                subquestion.search_query or subquestion.question, k=self.settings.top_k
            )
            for hit in hits:
                # Keep the best-scoring instance of any chunk seen more than once.
                current = state.evidence.get(hit.chunk_id)
                if current is None or hit.score > current.score:
                    state.evidence[hit.chunk_id] = hit

            self.memory.log_step(
                state.run_id,
                agent="retriever",
                action="retrieve",
                payload={"subquestion": subquestion.question, "k": self.settings.top_k},
                output={"chunk_ids": [h.chunk_id for h in hits]},
            )
            if not hits:
                logger.info("no evidence for %s", subquestion.id)
                continue

            result = self.analyst.run(
                run_id=state.run_id,
                question=state.question,
                subquestion=subquestion.question,
                evidence=hits,
                use_tools=use_tools,
            )
            state.usage = state.usage + result.usage
            state.tool_calls += result.tool_calls
            state.degraded |= not result.ok

            for finding in result.output or []:
                finding.subquestion_id = subquestion.id
                state.findings.append(finding)

        state.findings = _dedupe(state.findings)
        self.memory.scratch_set(
            state.run_id, "findings", [f.model_dump() for f in state.findings]
        )

    def _verify_citations(self, state: RunState) -> None:
        # In code, before the critic. A model checking its own ids isn't a check.
        known = set(state.evidence)
        kept: list[Finding] = []
        dropped = 0
        for finding in state.findings:
            valid = [c for c in finding.citations if c in known]
            if len(valid) != len(finding.citations):
                dropped += 1
            finding.citations = valid
            if valid:
                kept.append(finding)
        if dropped:
            logger.info(
                "run %s: stripped unresolvable citations from %d finding(s)",
                state.run_id, dropped,
            )
        self.memory.log_step(
            state.run_id,
            agent="verifier",
            action="verify_citations",
            output={
                "findings_in": len(state.findings),
                "findings_kept": len(kept),
                "citations_repaired": dropped,
            },
        )
        state.findings = kept

    def _synthesise_and_critique(self, state: RunState) -> None:
        notes = ""
        for attempt in range(self.settings.max_revisions + 1):
            result = self.synthesiser.run(
                run_id=state.run_id,
                question=state.question,
                findings=state.findings,
                revision_notes=notes,
            )
            state.usage = state.usage + result.usage
            state.tool_calls += result.tool_calls
            state.degraded |= not result.ok
            state.draft = result.output or {}

            claims = self._claims(state)
            if not claims:
                state.critique = Critique(
                    verdict="revise",
                    coverage=0.0,
                    issues=[
                        {
                            "claim": "(none)",
                            "issue": "no grounded claims were produced from the corpus",
                            "severity": "high",
                        }
                    ],
                )
                return

            critique_result = self.critic.run(
                run_id=state.run_id, claims=claims, evidence_ids=state.evidence_ids
            )
            state.usage = state.usage + critique_result.usage
            state.critique = critique_result.output
            state.degraded |= not critique_result.ok

            if state.critique.passed or attempt == self.settings.max_revisions:
                break

            state.revisions += 1
            notes = "\n".join(
                f"- [{i.severity}] {i.issue} (claim: {i.claim[:120]})"
                for i in state.critique.issues
            ) or "- The draft was not sufficiently grounded in cited evidence."
            logger.info(
                "run %s: revision %d requested (coverage %.2f)",
                state.run_id, state.revisions, state.critique.coverage,
            )

    # ---------- assembly ----------

    @staticmethod
    def _claims(state: RunState) -> list[dict[str, Any]]:
        sections = state.draft.get("sections") or []
        if sections:
            return [
                {"claim": s.get("body", ""), "citations": s.get("citations", [])}
                for s in sections
                if s.get("body")
            ]
        return [{"claim": f.claim, "citations": f.citations} for f in state.findings]

    def _assemble(self, state: RunState, started: float) -> ResearchBrief:
        sections = [
            Section(
                heading=s.get("heading", "Finding"),
                body=s.get("body", ""),
                citations=[c for c in s.get("citations", []) if c in state.evidence],
            )
            for s in state.draft.get("sections", [])
            if s.get("body")
        ]

        cited = {c for s in sections for c in s.citations}
        cited |= {c for f in state.findings for c in f.citations}
        citations = [
            Citation(
                chunk_id=cid,
                source=state.evidence[cid].chunk.citation_label(),
                snippet=state.evidence[cid].chunk.text,
            )
            for cid in sorted(cited)
            if cid in state.evidence
        ]

        return ResearchBrief(
            run_id=state.run_id,
            question=state.question,
            summary=state.draft.get("summary", ""),
            subquestions=state.subquestions,
            findings=state.findings,
            sections=sections,
            citations=citations,
            open_questions=state.draft.get("open_questions", []),
            critique=state.critique,
            stats=RunStats(
                subquestions=len(state.subquestions),
                chunks_retrieved=len(state.evidence),
                findings=len(state.findings),
                revisions=state.revisions,
                tool_calls=state.tool_calls,
                input_tokens=state.usage.input_tokens,
                output_tokens=state.usage.output_tokens,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
                provider=self.provider.name,
                degraded=state.degraded,
            ),
        )

    def _persist(self, state: RunState, brief: ResearchBrief) -> None:
        """Commit high-confidence grounded findings to durable memory."""
        if not brief.critique.passed:
            return
        for finding in brief.findings:
            if finding.confidence >= 0.6 and finding.citations:
                self.memory.remember(
                    finding.claim,
                    kind="finding",
                    citations=finding.citations,
                    confidence=finding.confidence,
                    run_id=state.run_id,
                )


def _dedupe(findings: list[Finding]) -> list[Finding]:
    """Sub-questions overlap, so the same passage - and the same claim - shows
    up more than once. Keep one, merge the citations."""
    merged: dict[str, Finding] = {}
    for finding in findings:
        key = " ".join(finding.claim.lower().split())
        existing = merged.get(key)
        if existing is None:
            merged[key] = finding
            continue
        existing.citations = sorted(set(existing.citations) | set(finding.citations))
        existing.confidence = max(existing.confidence, finding.confidence)
    return list(merged.values())
