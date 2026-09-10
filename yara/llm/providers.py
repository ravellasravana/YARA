"""Anthropic, OpenAI, Gemini.

Written against the HTTP APIs directly rather than pulling in three SDKs.
Smaller dependency surface, and - the part that actually matters - one retry
and error path instead of inheriting three different opinions about what
counts as a transient failure.
"""

from __future__ import annotations

import json
import logging
import random
import time
import uuid
from collections.abc import Iterable
from typing import Any

import httpx

from .base import (
    LLMError,
    LLMResponse,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
    split_system,
)

logger = logging.getLogger("yara.llm")

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class _Transient(Exception):
    """Internal marker for a failure worth retrying."""


class _HTTPProviderBase:
    name = "http"

    def __init__(self, *, api_key: str, model: str, timeout: float, max_retries: int):
        if not api_key:
            raise LLMError(f"{self.name} provider requires an API key")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.Client(timeout=timeout)

    def _post(self, url: str, *, headers: dict, payload: dict) -> dict[str, Any]:
        """POST, retrying transient failures only.

        First version retried everything that wasn't a 200, which meant a 400
        - bad schema, wrong model name, rejected key - got tried three times
        before surfacing. Burns quota and delays the error that tells you
        what's actually wrong. Now only the retryable statuses come back.
        """
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.post(url, headers=headers, json=payload)
                if response.status_code >= 400:
                    detail = f"{response.status_code}: {response.text[:300]}"
                    if response.status_code not in RETRYABLE_STATUS:
                        raise LLMError(f"{self.name} request rejected, {detail}")
                    raise _Transient(detail)
                return response.json()
            except LLMError:
                raise
            except (_Transient, httpx.HTTPError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt == self.max_retries - 1:
                    break
                backoff = (2**attempt) + random.random()
                logger.warning(
                    "%s call failed (attempt %d/%d), retrying in %.1fs: %s",
                    self.name,
                    attempt + 1,
                    self.max_retries,
                    backoff,
                    exc,
                )
                time.sleep(backoff)
        raise LLMError(f"{self.name} request failed: {last_error}") from last_error

    def close(self) -> None:
        self._client.close()


class AnthropicProvider(_HTTPProviderBase):
    name = "anthropic"
    URL = "https://api.anthropic.com/v1/messages"

    def complete(
        self,
        messages: Iterable[Message],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        system, convo = split_system(list(messages))
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or 2048,
            "messages": [_to_anthropic_message(m) for m in convo],
        }
        if system:
            payload["system"] = system
        if temperature is not None:
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = [t.to_anthropic() for t in tools]

        data = self._post(
            self.URL,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            payload=payload,
        )

        text_parts, calls = [], []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block["text"])
            elif block.get("type") == "tool_use":
                calls.append(
                    ToolCall(
                        id=block["id"], name=block["name"], arguments=block.get("input", {})
                    )
                )
        usage = data.get("usage", {})
        return LLMResponse(
            text="\n".join(text_parts).strip(),
            tool_calls=calls,
            usage=Usage(usage.get("input_tokens", 0), usage.get("output_tokens", 0)),
            stop_reason=data.get("stop_reason", "end_turn"),
            raw=data,
        )


def _to_anthropic_message(m: Message) -> dict[str, Any]:
    if m.role == "tool":
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": r.call_id,
                    "content": r.content,
                    "is_error": r.is_error,
                }
                for r in m.tool_results
            ],
        }
    if m.role == "assistant" and m.tool_calls:
        content: list[dict[str, Any]] = []
        if m.content:
            content.append({"type": "text", "text": m.content})
        content += [
            {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
            for c in m.tool_calls
        ]
        return {"role": "assistant", "content": content}
    return {"role": m.role, "content": m.content}


class OpenAIProvider(_HTTPProviderBase):
    name = "openai"
    URL = "https://api.openai.com/v1/chat/completions"

    def complete(
        self,
        messages: Iterable[Message],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _to_openai_messages(list(messages)),
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = [t.to_openai() for t in tools]

        data = self._post(
            self.URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            payload=payload,
        )
        choice = data["choices"][0]
        message = choice["message"]
        calls = [
            ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=json.loads(tc["function"].get("arguments") or "{}"),
            )
            for tc in message.get("tool_calls") or []
        ]
        usage = data.get("usage", {})
        return LLMResponse(
            text=(message.get("content") or "").strip(),
            tool_calls=calls,
            usage=Usage(
                usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
            ),
            stop_reason=choice.get("finish_reason", "stop"),
            raw=data,
        )


def _to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            out += [
                {"role": "tool", "tool_call_id": r.call_id, "content": r.content}
                for r in m.tool_results
            ]
        elif m.role == "assistant" and m.tool_calls:
            out.append(
                {
                    "role": "assistant",
                    "content": m.content or None,
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.name,
                                "arguments": json.dumps(c.arguments),
                            },
                        }
                        for c in m.tool_calls
                    ],
                }
            )
        else:
            out.append({"role": m.role, "content": m.content})
    return out


class GeminiProvider(_HTTPProviderBase):
    name = "gemini"
    BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def complete(
        self,
        messages: Iterable[Message],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        system, convo = split_system(list(messages))
        payload: dict[str, Any] = {"contents": _to_gemini_contents(convo)}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        gen: dict[str, Any] = {}
        if temperature is not None:
            gen["temperature"] = temperature
        if max_tokens:
            gen["maxOutputTokens"] = max_tokens
        if gen:
            payload["generationConfig"] = gen
        if tools:
            payload["tools"] = [
                {"functionDeclarations": [t.to_gemini() for t in tools]}
            ]

        data = self._post(
            f"{self.BASE}/{self.model}:generateContent?key={self.api_key}",
            headers={"Content-Type": "application/json"},
            payload=payload,
        )
        candidates = data.get("candidates") or []
        if not candidates:
            raise LLMError(f"gemini returned no candidates: {data}")
        parts = candidates[0].get("content", {}).get("parts", [])

        text_parts, calls = [], []
        for part in parts:
            if "text" in part:
                text_parts.append(part["text"])
            elif "functionCall" in part:
                fc = part["functionCall"]
                calls.append(
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:12]}",
                        name=fc["name"],
                        arguments=fc.get("args", {}),
                    )
                )
        usage = data.get("usageMetadata", {})
        return LLMResponse(
            text="\n".join(text_parts).strip(),
            tool_calls=calls,
            usage=Usage(
                usage.get("promptTokenCount", 0), usage.get("candidatesTokenCount", 0)
            ),
            stop_reason=candidates[0].get("finishReason", "STOP"),
            raw=data,
        )


def _to_gemini_contents(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            out.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": r.name,
                                "response": {"result": r.content},
                            }
                        }
                        for r in m.tool_results
                    ],
                }
            )
        elif m.role == "assistant":
            parts: list[dict[str, Any]] = []
            if m.content:
                parts.append({"text": m.content})
            parts += [
                {"functionCall": {"name": c.name, "args": c.arguments}}
                for c in m.tool_calls
            ]
            out.append({"role": "model", "parts": parts or [{"text": ""}]})
        else:
            out.append({"role": "user", "parts": [{"text": m.content}]})
    return out
