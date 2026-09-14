"""What every agent shares: prompt building and the tool-calling loop.

The loop is capped. A model that keeps asking for tools has no stopping
condition of its own, and an uncapped loop is the standard way an agent
system ends up with an unbounded bill.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ..llm.base import LLMProvider, LLMResponse, Message, Usage, parse_json_loose
from ..memory.store import MemoryStore
from ..tools.base import ToolRegistry

logger = logging.getLogger("yara.agents")


@dataclass
class AgentResult:
    output: Any
    text: str = ""
    usage: Usage = field(default_factory=Usage)
    tool_calls: int = 0
    iterations: int = 1
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class BaseAgent:
    role: str = "generic"
    name: str = "agent"
    expects_json: bool = True

    def __init__(
        self,
        provider: LLMProvider,
        *,
        tools: ToolRegistry | None = None,
        memory: MemoryStore | None = None,
        max_iterations: int = 6,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ):
        self.provider = provider
        self.tools = tools
        self.memory = memory
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ---------- to override ----------

    def system_prompt(self) -> str:
        raise NotImplementedError

    def build_prompt(self, **kwargs: Any) -> str:
        raise NotImplementedError

    def parse(self, response: LLMResponse) -> Any:
        if not self.expects_json:
            return response.text
        return parse_json_loose(response.text)

    # ---------- shared machinery ----------

    def run(self, *, run_id: str | None = None, use_tools: bool = False, **kwargs: Any) -> AgentResult:
        started = time.perf_counter()
        system = f"YARA_ROLE: {self.role}\n\n{self.system_prompt()}"
        messages: list[Message] = [
            Message.system(system),
            Message.user(self.build_prompt(**kwargs)),
        ]
        specs = self.tools.specs() if (use_tools and self.tools) else None

        usage = Usage()
        tool_calls = 0
        response: LLMResponse | None = None
        iterations = 0

        for iterations in range(1, self.max_iterations + 1):  # noqa: B007 - read after loop
            response = self.provider.complete(
                messages,
                tools=specs,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            usage = usage + response.usage

            if not response.wants_tools:
                break

            messages.append(Message.assistant(response.text, response.tool_calls))
            results = [self.tools.dispatch(call) for call in response.tool_calls]
            tool_calls += len(results)
            messages.append(Message.tool(results))
        else:
            logger.warning(
                "%s hit the %d-iteration tool budget; answering with what it has",
                self.name, self.max_iterations,
            )
            messages.append(
                Message.user(
                    "Tool budget exhausted. Answer now using the information already "
                    "gathered, in the required JSON format."
                )
            )
            response = self.provider.complete(
                messages, temperature=self.temperature, max_tokens=self.max_tokens
            )
            usage = usage + response.usage

        elapsed = (time.perf_counter() - started) * 1000
        try:
            output = self.parse(response)
            result = AgentResult(
                output=output,
                text=response.text,
                usage=usage,
                tool_calls=tool_calls,
                iterations=iterations,
            )
        except ValueError as exc:
            logger.error("%s produced unparseable output: %s", self.name, exc)
            result = AgentResult(
                output=self.fallback(**kwargs),
                text=response.text if response else "",
                usage=usage,
                tool_calls=tool_calls,
                iterations=iterations,
                error=str(exc),
            )

        if self.memory and run_id:
            self.memory.log_step(
                run_id,
                agent=self.name,
                action="complete",
                payload={k: _truncate(v) for k, v in kwargs.items()},
                output={
                    "parsed": _truncate(result.output),
                    "error": result.error,
                    "tool_calls": tool_calls,
                },
                latency_ms=elapsed,
            )
        return result

    def fallback(self, **kwargs: Any) -> Any:
        # What to hand back when the model's output won't parse.
        return {}


def _truncate(value: Any, limit: int = 2000) -> Any:
    text = value if isinstance(value, str) else repr(value)
    return text[:limit] + ("..." if len(text) > limit else "")
