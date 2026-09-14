"""The four agents. One narrow job each.

One prompt asked to plan, read, write and check itself does all four badly,
and gives you no seam where grounding can be checked separately from the
writing that produced it.
"""

from __future__ import annotations

import json
from typing import Any

from ..briefs.schema import Critique, Finding, SubQuestion
from ..llm.base import LLMResponse
from ..retrieval.retriever import ScoredChunk
from .base import BaseAgent

# --------------------------------------------------------------------------
# Planner
# --------------------------------------------------------------------------


class PlannerAgent(BaseAgent):
    role = "planner"
    name = "planner"

    def system_prompt(self) -> str:
        return (
            "You decompose a research question into independent sub-questions that "
            "can each be answered from retrieved source passages.\n\n"
            "Rules:\n"
            "- Cover distinct angles. Do not restate the same question twice.\n"
            "- Each sub-question must be answerable from documents, not opinion.\n"
            "- Write search_query as keywords a retrieval system would match, not prose.\n\n"
            'Respond with JSON only: {"subquestions": [{"id": "sq1", "question": "...", '
            '"angle": "definition|evidence|mechanism|limitation|comparison", '
            '"search_query": "..."}]}'
        )

    def build_prompt(self, *, question: str, max_subquestions: int = 5, prior: str = "") -> str:
        parts = [
            f"<QUESTION>{question}</QUESTION>",
            f"<MAX_SUBQUESTIONS>{max_subquestions}</MAX_SUBQUESTIONS>",
        ]
        if prior:
            parts.append(
                f"<PRIOR_KNOWLEDGE>{prior}</PRIOR_KNOWLEDGE>\n"
                "Prefer sub-questions that go beyond what prior knowledge already covers."
            )
        return "\n\n".join(parts)

    def parse(self, response: LLMResponse) -> list[SubQuestion]:
        data = response.json()
        raw = data.get("subquestions", data) if isinstance(data, dict) else data
        out: list[SubQuestion] = []
        for i, item in enumerate(raw or []):
            if not isinstance(item, dict) or not item.get("question"):
                continue
            out.append(
                SubQuestion(
                    id=item.get("id") or f"sq{i + 1}",
                    question=item["question"],
                    angle=item.get("angle", "general"),
                    search_query=item.get("search_query", ""),
                )
            )
        return out

    def fallback(self, *, question: str = "", **_: Any) -> list[SubQuestion]:
        # One sub-question is still a plan. Better than nothing.
        return [SubQuestion(id="sq1", question=question, angle="general", search_query=question)]


# --------------------------------------------------------------------------
# Analyst
# --------------------------------------------------------------------------


class AnalysisAgent(BaseAgent):
    role = "analysis"
    name = "analyst"

    def system_prompt(self) -> str:
        return (
            "You extract factual findings from retrieved passages.\n\n"
            "Rules:\n"
            "- Every claim must be supported by at least one passage you were given.\n"
            "- Cite by the exact chunk id shown in square brackets before each passage.\n"
            "- Never introduce information not present in the passages.\n"
            "- If the passages do not answer the sub-question, return an empty list.\n"
            "- confidence is 0.0-1.0 and reflects how directly the passage supports the claim.\n\n"
            'Respond with JSON only: {"findings": [{"claim": "...", '
            '"citations": ["chunk_id"], "confidence": 0.0}]}'
        )

    def build_prompt(
        self, *, subquestion: str, evidence: list[ScoredChunk], question: str = "", **_: Any
    ) -> str:
        blocks = "\n\n".join(
            f"[{sc.chunk_id}] ({sc.chunk.citation_label()})\n{sc.chunk.text}" for sc in evidence
        )
        return (
            f"<QUESTION>{question}</QUESTION>\n\n"
            f"<SUBQUESTION>{subquestion}</SUBQUESTION>\n\n"
            f"<EVIDENCE>\n{blocks}\n</EVIDENCE>"
        )

    def parse(self, response: LLMResponse) -> list[Finding]:
        data = response.json()
        raw = data.get("findings", data) if isinstance(data, dict) else data
        out: list[Finding] = []
        for item in raw or []:
            if not isinstance(item, dict) or not item.get("claim"):
                continue
            citations = item.get("citations") or []
            if isinstance(citations, str):
                citations = [citations]
            out.append(
                Finding(
                    claim=str(item["claim"]).strip(),
                    citations=[str(c) for c in citations],
                    confidence=float(item.get("confidence", 0.5)),
                )
            )
        return out

    def fallback(self, **_: Any) -> list[Finding]:
        return []


# --------------------------------------------------------------------------
# Synthesiser
# --------------------------------------------------------------------------


class SynthesisAgent(BaseAgent):
    role = "synthesis"
    name = "synthesiser"

    def system_prompt(self) -> str:
        return (
            "You turn verified findings into a research brief.\n\n"
            "Rules:\n"
            "- Use only the findings supplied. Add no new facts.\n"
            "- Carry forward the citations attached to each finding you use.\n"
            "- Group related findings under a heading rather than listing them flat.\n"
            "- Where findings conflict, say so explicitly instead of picking one.\n\n"
            'Respond with JSON only: {"summary": "...", "sections": [{"heading": "...", '
            '"body": "...", "citations": ["chunk_id"]}], "open_questions": ["..."]}'
        )

    def build_prompt(
        self, *, question: str, findings: list[Finding], revision_notes: str = "", **_: Any
    ) -> str:
        payload = json.dumps(
            [
                {"claim": f.claim, "citations": f.citations, "confidence": f.confidence}
                for f in findings
            ],
            indent=2,
        )
        prompt = f"<QUESTION>{question}</QUESTION>\n\n<FINDINGS>\n{payload}\n</FINDINGS>"
        if revision_notes:
            prompt += (
                f"\n\n<REVISION_NOTES>\n{revision_notes}\n</REVISION_NOTES>\n"
                "A previous draft was rejected for the reasons above. Fix them. "
                "Drop any claim you cannot cite."
            )
        return prompt

    def parse(self, response: LLMResponse) -> dict[str, Any]:
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("synthesis output must be a JSON object")
        return {
            "summary": str(data.get("summary", "")).strip(),
            "sections": [s for s in data.get("sections", []) if isinstance(s, dict)],
            "open_questions": [str(q) for q in data.get("open_questions", [])],
        }

    def fallback(self, *, findings: list[Finding] | None = None, **_: Any) -> dict[str, Any]:
        findings = findings or []
        return {
            "summary": " ".join(f.claim for f in findings[:3]),
            "sections": [
                {"heading": f"Finding {i + 1}", "body": f.claim, "citations": f.citations}
                for i, f in enumerate(findings[:6])
            ],
            "open_questions": [],
        }


# --------------------------------------------------------------------------
# Critic
# --------------------------------------------------------------------------


class CriticAgent(BaseAgent):
    role = "critic"
    name = "critic"

    def system_prompt(self) -> str:
        return (
            "You audit a draft brief for grounding. You are not editing for style.\n\n"
            "Flag a claim when it:\n"
            "- carries no citation;\n"
            "- cites a chunk id that is not in the evidence id list;\n"
            "- asserts more than the cited passage supports;\n"
            "- contradicts another claim without acknowledging the conflict.\n\n"
            'Respond with JSON only: {"verdict": "pass"|"revise", "coverage": 0.0, '
            '"issues": [{"claim": "...", "issue": "...", "severity": "low|medium|high"}]}'
        )

    def build_prompt(self, *, claims: list[dict[str, Any]], evidence_ids: list[str], **_: Any) -> str:
        return (
            f"<CLAIMS>\n{json.dumps(claims, indent=2)}\n</CLAIMS>\n\n"
            f"<EVIDENCE_IDS>{', '.join(evidence_ids)}</EVIDENCE_IDS>"
        )

    def parse(self, response: LLMResponse) -> Critique:
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("critique output must be a JSON object")
        return Critique(
            verdict=data.get("verdict", "pass"),
            coverage=float(data.get("coverage", 0.0)),
            issues=[i for i in data.get("issues", []) if isinstance(i, dict) and i.get("claim")],
        )

    def fallback(self, **_: Any) -> Critique:
        # Unparseable critique is not approval. Fail closed.
        return Critique(verdict="revise", coverage=0.0, issues=[])
