"""LLM abstraction layer: one interface, four interchangeable backends."""

from __future__ import annotations

import logging

from ..config import Settings, get_settings
from .base import (
    LLMError,
    LLMProvider,
    LLMResponse,
    Message,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
    parse_json_loose,
)
from .echo import EchoProvider
from .providers import AnthropicProvider, GeminiProvider, OpenAIProvider

logger = logging.getLogger("yara.llm")

_KEY_FIELD = {
    "anthropic": "anthropic_api_key",
    "gemini": "gemini_api_key",
    "openai": "openai_api_key",
}
_CLASSES = {
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
}


def get_provider(settings: Settings | None = None, *, name: str | None = None) -> LLMProvider:
    """Build the configured provider, degrading to `echo` when no key is set."""
    settings = settings or get_settings()
    name = (name or settings.provider).lower()

    if name == "echo":
        return EchoProvider(model=settings.model)

    if name not in _CLASSES:
        raise LLMError(f"unknown provider {name!r}; expected one of {sorted(_CLASSES) + ['echo']}")

    api_key = getattr(settings, _KEY_FIELD[name], None)
    if not api_key:
        logger.warning(
            "provider %r selected but no API key found; falling back to deterministic echo provider",
            name,
        )
        return EchoProvider(model=settings.model)

    return _CLASSES[name](
        api_key=api_key,
        model=settings.model,
        timeout=settings.request_timeout,
        max_retries=settings.max_retries,
    )


__all__ = [
    "AnthropicProvider", "EchoProvider", "GeminiProvider", "OpenAIProvider",
    "LLMError", "LLMProvider", "LLMResponse", "Message", "ToolCall",
    "ToolResult", "ToolSpec", "Usage", "get_provider", "parse_json_loose",
]
