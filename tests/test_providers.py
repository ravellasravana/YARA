"""Vendor translation and HTTP behaviour, tested against a mock transport.

Translation is where a silent bug hides: a mistranslated tool result doesn't
raise, it just makes the model behave as though the tool never ran.
"""

import httpx
import pytest

from yara.config import Settings
from yara.llm import get_provider
from yara.llm.base import LLMError, Message, ToolCall, ToolResult
from yara.llm.providers import (
    AnthropicProvider,
    GeminiProvider,
    OpenAIProvider,
    _to_anthropic_message,
    _to_gemini_contents,
    _to_openai_messages,
)


class TestMessageTranslation:
    def test_anthropic_turns_a_tool_result_into_a_user_turn(self):
        payload = _to_anthropic_message(Message.tool([ToolResult("call_1", "search", "found it")]))
        assert payload["role"] == "user"
        assert payload["content"][0]["type"] == "tool_result"
        assert payload["content"][0]["tool_use_id"] == "call_1"

    def test_anthropic_interleaves_text_and_tool_use(self):
        payload = _to_anthropic_message(
            Message.assistant("thinking", [ToolCall("c1", "search", {"q": "x"})])
        )
        assert [b["type"] for b in payload["content"]] == ["text", "tool_use"]

    def test_openai_splits_tool_results_into_separate_turns(self):
        out = _to_openai_messages(
            [Message.tool([ToolResult("c1", "s", "a"), ToolResult("c2", "s", "b")])]
        )
        assert len(out) == 2
        assert all(m["role"] == "tool" for m in out)

    def test_openai_serialises_arguments_as_a_json_string(self):
        out = _to_openai_messages([Message.assistant("", [ToolCall("c1", "search", {"q": "x"})])])
        assert out[0]["tool_calls"][0]["function"]["arguments"] == '{"q": "x"}'

    def test_gemini_calls_the_assistant_role_model(self):
        assert _to_gemini_contents([Message.assistant("hello")])[0]["role"] == "model"

    def test_gemini_never_emits_empty_parts(self):
        """An empty parts array gets rejected by the API."""
        assert _to_gemini_contents([Message.assistant("")])[0]["parts"]

    def test_gemini_tool_result_uses_function_response(self):
        out = _to_gemini_contents([Message.tool([ToolResult("c1", "search", "r")])])
        assert out[0]["parts"][0]["functionResponse"]["name"] == "search"


def _provider(cls, handler):
    p = cls(api_key="test-key", model="m", timeout=5, max_retries=3)
    p._client = httpx.Client(transport=httpx.MockTransport(handler))
    return p


class TestResponseParsing:
    def test_anthropic_separates_text_from_tool_calls(self):
        def handler(_):
            return httpx.Response(200, json={
                "content": [
                    {"type": "text", "text": "sure"},
                    {"type": "tool_use", "id": "c1", "name": "search", "input": {"q": "x"}},
                ],
                "usage": {"input_tokens": 11, "output_tokens": 7},
                "stop_reason": "tool_use",
            })

        r = _provider(AnthropicProvider, handler).complete([Message.user("hi")])
        assert r.text == "sure"
        assert r.wants_tools
        assert r.tool_calls[0].arguments == {"q": "x"}
        assert r.usage.input_tokens == 11

    def test_openai_parses_tool_call_arguments(self):
        def handler(_):
            return httpx.Response(200, json={
                "choices": [{
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {"id": "c1", "function": {"name": "search", "arguments": '{"q": "x"}'}}
                        ],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            })

        r = _provider(OpenAIProvider, handler).complete([Message.user("hi")])
        assert r.tool_calls[0].arguments == {"q": "x"}
        assert r.text == ""

    def test_gemini_parses_a_function_call(self):
        def handler(_):
            return httpx.Response(200, json={
                "candidates": [{
                    "content": {"parts": [{"functionCall": {"name": "search", "args": {"q": "x"}}}]},
                    "finishReason": "STOP",
                }],
                "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 1},
            })

        assert _provider(GeminiProvider, handler).complete([Message.user("hi")]).tool_calls[0].name == "search"

    def test_gemini_with_no_candidates_raises(self):
        p = _provider(GeminiProvider, lambda _: httpx.Response(200, json={"candidates": []}))
        with pytest.raises(LLMError):
            p.complete([Message.user("hi")])


class TestRetries:
    def test_retries_a_503_then_succeeds(self, monkeypatch):
        monkeypatch.setattr("yara.llm.providers.time.sleep", lambda _: None)
        n = {"calls": 0}

        def handler(_):
            n["calls"] += 1
            if n["calls"] < 3:
                return httpx.Response(503, text="unavailable")
            return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}]})

        assert _provider(AnthropicProvider, handler).complete([Message.user("hi")]).text == "ok"
        assert n["calls"] == 3

    def test_gives_up_after_max_retries(self, monkeypatch):
        monkeypatch.setattr("yara.llm.providers.time.sleep", lambda _: None)
        p = _provider(AnthropicProvider, lambda _: httpx.Response(500, text="boom"))
        with pytest.raises(LLMError):
            p.complete([Message.user("hi")])

    def test_does_not_retry_a_400(self, monkeypatch):
        """Bad schema or wrong model name. Retrying burns quota and hides it."""
        monkeypatch.setattr("yara.llm.providers.time.sleep", lambda _: None)
        n = {"calls": 0}

        def handler(_):
            n["calls"] += 1
            return httpx.Response(400, text="bad request")

        with pytest.raises(LLMError):
            _provider(AnthropicProvider, handler).complete([Message.user("hi")])
        assert n["calls"] == 1


class TestProviderSelection:
    def test_no_key_falls_back_to_echo(self):
        """Degrade at startup, not at request time."""
        assert get_provider(Settings(provider="anthropic", anthropic_api_key=None)).name == "echo"

    def test_builds_the_named_provider_when_a_key_is_present(self):
        assert get_provider(Settings(provider="anthropic", anthropic_api_key="sk-test")).name == "anthropic"

    def test_unknown_provider_is_rejected(self):
        with pytest.raises(LLMError):
            get_provider(Settings(), name="not-a-provider")
