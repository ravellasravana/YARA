"""Tools the model can call, and the registry that dispatches them.

A tool is a named callable with a JSON Schema for its arguments. The registry
hands specs to whichever provider is active and runs whatever the model asks
for. A tool that blows up returns an error string the model can react to -
it never takes the run down with it.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from ..llm.base import ToolCall, ToolResult, ToolSpec

logger = logging.getLogger("yara.tools")


class ToolError(Exception):
    """Raised inside a tool to signal a recoverable failure to the model."""


class Tool(ABC):
    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {"type": "object", "properties": {}}

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """Execute the tool. Return anything JSON-serialisable."""

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name, description=self.description, parameters=self.parameters
        )

    def validate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        # Models sometimes hallucinate extra kwargs. Drop them rather than fail.
        schema = self.parameters or {}
        properties = schema.get("properties", {})
        missing = [r for r in schema.get("required", []) if r not in arguments]
        if missing:
            raise ToolError(f"missing required argument(s): {', '.join(missing)}")
        unknown = set(arguments) - set(properties)
        if unknown:
            logger.debug("tool %s ignoring unknown args: %s", self.name, sorted(unknown))
        return {k: v for k, v in arguments.items() if k in properties}


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError(f"{type(tool).__name__} must define a name")
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values()]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def dispatch(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content=f"Unknown tool {call.name!r}. Available: {', '.join(self.names())}",
                is_error=True,
            )
        try:
            arguments = tool.validate(call.arguments or {})
            output = tool.run(**arguments)
            content = output if isinstance(output, str) else json.dumps(output, default=str)
            return ToolResult(call_id=call.id, name=call.name, content=content)
        except ToolError as exc:
            return ToolResult(call_id=call.id, name=call.name, content=str(exc), is_error=True)
        except Exception as exc:  # noqa: BLE001 - a tool bug must not kill the run
            logger.exception("tool %s raised", call.name)
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content=f"{type(exc).__name__}: {exc}",
                is_error=True,
            )
