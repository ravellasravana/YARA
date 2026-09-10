"""LLM layer. One interface, swappable backends."""

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

__all__ = [
    "EchoProvider", "LLMError", "LLMProvider", "LLMResponse", "Message",
    "ToolCall", "ToolResult", "ToolSpec", "Usage", "parse_json_loose",
]
