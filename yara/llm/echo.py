"""A provider that doesn't call anything. Deterministic, no key, no network.

Not a mock. It's extractive - synthesis pulls the highest-scoring sentences
straight out of the retrieved evidence - which makes it useful three ways:

- CI runs the whole pipeline without a key, so a red build means the
  orchestration changed rather than a model drifted.
- It can't hallucinate, since every sentence it emits appears verbatim in a
  source. That's a floor to measure the real thing against, and a system that
  doesn't beat it is paying for generation and getting nothing.
- It's the fallback when a hosted provider is down. Degraded output beats an
  exception.

Agents put a YARA_ROLE line in their system prompt and this branches on it.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from typing import Any

from .base import LLMResponse, Message, ToolSpec, Usage, split_system

_ROLE_RE = re.compile(r"YARA_ROLE:\s*([a-z_]+)", re.IGNORECASE)
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "were",
    "for", "on", "with", "as", "by", "that", "this", "it", "be", "from", "at",
    "which", "how", "what", "why", "does", "do", "can", "will", "has", "have",
}


class EchoProvider:
    name = "echo"

    def __init__(self, *, model: str = "echo-extractive-v1", **_: Any):
        self.model = model

    def complete(
        self,
        messages: Iterable[Message],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        msgs = list(messages)
        system, convo = split_system(msgs)
        role = self._role(system)
        prompt = "\n\n".join(m.content for m in convo if m.content)

        handler = {
            "planner": self._plan,
            "analysis": self._analyse,
            "synthesis": self._synthesise,
            "critic": self._critique,
        }.get(role, self._passthrough)

        text = handler(prompt)
        return LLMResponse(
            text=text,
            usage=Usage(_approx_tokens(prompt), _approx_tokens(text)),
            stop_reason="end_turn",
            raw={"provider": "echo", "role": role, "model": self.model},
        )

    # ---------- role handlers ----------

    def _plan(self, prompt: str) -> str:
        question = _tagged(prompt, "QUESTION") or prompt
        limit = int(_tagged(prompt, "MAX_SUBQUESTIONS") or 3)
        terms = _keywords(question, limit=6)

        angles = [
            ("definition", "What is meant by {} and how is it defined?"),
            ("evidence", "What evidence or results are reported about {}?"),
            ("mechanism", "How does {} work or what drives it?"),
            ("limitation", "What limitations or open problems affect {}?"),
            ("comparison", "How does {} compare with alternatives?"),
        ]
        focus = " ".join(terms[:3]) or "the topic"
        subs = [
            {
                "id": f"sq{i + 1}",
                "question": template.format(focus),
                "angle": angle,
                "search_query": f"{focus} {angle}",
            }
            for i, (angle, template) in enumerate(angles[:limit])
        ]
        return json.dumps({"subquestions": subs}, indent=2)

    def _analyse(self, prompt: str) -> str:
        evidence = _parse_evidence(prompt)
        question = _tagged(prompt, "SUBQUESTION") or _tagged(prompt, "QUESTION") or ""
        terms = _keywords(question)

        findings = []
        for chunk_id, body in evidence:
            best = _best_sentence(body, terms)
            if not best:
                continue
            findings.append(
                {
                    "claim": best,
                    "citations": [chunk_id],
                    "confidence": round(min(0.95, 0.55 + 0.1 * _overlap(best, terms)), 2),
                }
            )
        findings.sort(key=lambda f: -f["confidence"])
        return json.dumps({"findings": findings[:5]}, indent=2)

    def _synthesise(self, prompt: str) -> str:
        question = _tagged(prompt, "QUESTION") or ""
        findings = _parse_findings(prompt)
        if not findings:
            return json.dumps(
                {
                    "summary": "No supporting evidence was retrieved for this question.",
                    "sections": [],
                    "open_questions": ["Ingest source material covering this topic."],
                },
                indent=2,
            )

        top = findings[:3]
        summary = " ".join(f["claim"].rstrip(".") + "." for f in top)
        sections = [
            {
                "heading": f"Finding {i + 1}",
                "body": f["claim"],
                "citations": f.get("citations", []),
            }
            for i, f in enumerate(findings[:6])
        ]
        return json.dumps(
            {
                "summary": summary[:1200],
                "sections": sections,
                "open_questions": [
                    f"What additional sources would strengthen the answer to: {question}"
                ],
            },
            indent=2,
        )

    def _critique(self, prompt: str) -> str:
        claims = _parse_claims(prompt)
        known = set(_parse_chunk_ids(prompt))
        issues = []
        for claim in claims:
            cites = claim.get("citations", [])
            if not cites:
                issues.append(
                    {"claim": claim.get("claim", "")[:160], "issue": "uncited", "severity": "high"}
                )
            elif known and not set(cites) & known:
                issues.append(
                    {
                        "claim": claim.get("claim", "")[:160],
                        "issue": "citation does not resolve to retrieved evidence",
                        "severity": "high",
                    }
                )
        grounded = len(claims) - len(issues)
        coverage = grounded / len(claims) if claims else 0.0
        return json.dumps(
            {
                "verdict": "pass" if coverage >= 0.8 and claims else "revise",
                "coverage": round(coverage, 3),
                "issues": issues,
            },
            indent=2,
        )

    def _passthrough(self, prompt: str) -> str:
        digest = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        head = _first_sentences(prompt, 2)
        return f"[echo:{digest}] {head}".strip()

    @staticmethod
    def _role(system: str) -> str:
        match = _ROLE_RE.search(system or "")
        return match.group(1).lower() if match else "generic"


# ---------- helpers ----------


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _tagged(prompt: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", prompt, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def _keywords(text: str, limit: int = 10) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", (text or "").lower())
    seen: dict[str, int] = {}
    for w in words:
        if w in _STOPWORDS:
            continue
        seen[w] = seen.get(w, 0) + 1
    return [w for w, _ in sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))][:limit]


def _overlap(sentence: str, terms: list[str]) -> int:
    low = sentence.lower()
    return sum(1 for t in terms if t in low)


def _best_sentence(body: str, terms: list[str]) -> str | None:
    sentences = [s.strip() for s in _SENT_RE.split(body) if len(s.strip()) > 40]
    if not sentences:
        stripped = body.strip()
        return stripped[:300] if stripped else None
    scored = sorted(
        sentences, key=lambda s: (-_overlap(s, terms), sentences.index(s))
    )
    return scored[0][:400]


def _first_sentences(text: str, n: int) -> str:
    return " ".join(_SENT_RE.split(text.strip())[:n])[:400]


def _parse_evidence(prompt: str) -> list[tuple[str, str]]:
    """Pull `[chunk_id] (label)\\n body` blocks out of an <EVIDENCE> section.

    The parenthesised source label is dropped so it never leaks into an
    extracted claim.
    """
    block = _tagged(prompt, "EVIDENCE") or ""
    matches = re.findall(r"\[([^\]\s]+)\]\s*(.*?)(?=\n\[[^\]\s]+\]|\Z)", block, re.DOTALL)
    out = []
    for chunk_id, body in matches:
        body = re.sub(r"^\s*\([^)\n]*\)\s*\n?", "", body).strip()
        if body:
            out.append((chunk_id, body))
    return out


def _parse_findings(prompt: str) -> list[dict[str, Any]]:
    block = _tagged(prompt, "FINDINGS")
    if not block:
        return []
    try:
        data = json.loads(block)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = data.get("findings", [])
    return [f for f in data if isinstance(f, dict) and f.get("claim")]


def _parse_claims(prompt: str) -> list[dict[str, Any]]:
    block = _tagged(prompt, "CLAIMS")
    if not block:
        return []
    try:
        data = json.loads(block)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else data.get("claims", [])


def _parse_chunk_ids(prompt: str) -> list[str]:
    block = _tagged(prompt, "EVIDENCE_IDS") or ""
    return [x.strip() for x in block.replace("\n", ",").split(",") if x.strip()]
