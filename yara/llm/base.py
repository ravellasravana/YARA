"""One interface, several vendors behind it.

Everything normalises to the same Message/LLMResponse shape so nothing above
this layer learns a vendor SDK. Tool schemas are written once in JSON Schema
and translated per provider.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ToolSpec:
    """A callable exposed to the model, described in JSON Schema."""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_anthropic(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_gemini(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": _strip_unsupported(self.parameters),
        }


def _strip_unsupported(schema: dict[str, Any]) -> dict[str, Any]:
    """Gemini 400s on a few JSON Schema keywords. Drop them rather than keep a
    second copy of every schema."""
    banned = {"additionalProperties", "$schema", "default"}
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key in banned:
            continue
        if isinstance(value, dict):
            out[key] = _strip_unsupported(value)
        elif isinstance(value, list):
            out[key] = [
                _strip_unsupported(v) if isinstance(v, dict) else v for v in value
            ]
        else:
            out[key] = value
    return out


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)

    @staticmethod
    def system(content: str) -> Message:
        return Message(role="system", content=content)

    @staticmethod
    def user(content: str) -> Message:
        return Message(role="user", content=content)

    @staticmethod
    def assistant(content: str = "", tool_calls: list[ToolCall] | None = None) -> Message:
        return Message(role="assistant", content=content, tool_calls=tool_calls or [])

    @staticmethod
    def tool(results: list[ToolResult]) -> Message:
        return Message(role="tool", tool_results=results)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = "end_turn"
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)

    def json(self) -> Any:
        """Parse the response body as JSON, tolerating fenced code blocks."""
        return parse_json_loose(self.text)


class LLMError(RuntimeError):
    """Raised when a provider call fails after exhausting retries."""


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def complete(
        self,
        messages: Iterable[Message],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...


def parse_json_loose(text: str) -> Any:
    """Dig JSON out of whatever the model actually returned.

    Asking for JSON only gets you JSON most of the time. The rest of the time
    it's fenced, or wrapped in a sentence. Handles bare, fenced, and embedded.
    Raises ValueError if there's genuinely nothing parseable.
    """
    if text is None:
        raise ValueError("no content to parse")
    candidate = text.strip()

    if candidate.startswith("```"):
        candidate = candidate.split("```")[1] if "```" in candidate[3:] else candidate[3:]
        if candidate.lstrip().lower().startswith("json"):
            candidate = candidate.lstrip()[4:]
        candidate = candidate.strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        end = candidate.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue

    raise ValueError(f"could not parse JSON from response: {text[:200]!r}")


def split_system(messages: list[Message]) -> tuple[str, list[Message]]:
    """Anthropic and Gemini take the system prompt out of band, OpenAI doesn't."""
    system_parts = [m.content for m in messages if m.role == "system" and m.content]
    rest = [m for m in messages if m.role != "system"]
    return "\n\n".join(system_parts), rest
