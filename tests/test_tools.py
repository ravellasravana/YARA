import pytest

from yara.llm.base import ToolCall
from yara.tools.base import Tool, ToolError, ToolRegistry
from yara.tools.builtin import CalculatorTool, RecallMemoryTool, SearchCorpusTool


class Boom(Tool):
    name = "boom"
    description = "always raises"
    parameters = {"type": "object", "properties": {}}

    def run(self, **_):
        raise RuntimeError("kaboom")


class TestRegistry:
    def test_dispatch_success(self, retriever):
        registry = ToolRegistry([SearchCorpusTool(retriever)])
        result = registry.dispatch(ToolCall("1", "search_corpus", {"query": "grounding"}))
        assert not result.is_error and "chunk_id" in result.content

    def test_unknown_tool_is_an_error_not_a_crash(self):
        result = ToolRegistry([CalculatorTool()]).dispatch(ToolCall("1", "nope", {}))
        assert result.is_error and "Unknown tool" in result.content

    def test_tool_exception_is_contained(self):
        """A broken tool must return an error the model can react to, not kill the run."""
        result = ToolRegistry([Boom()]).dispatch(ToolCall("1", "boom", {}))
        assert result.is_error and "kaboom" in result.content

    def test_missing_required_argument_reported(self):
        result = ToolRegistry([CalculatorTool()]).dispatch(ToolCall("1", "calculator", {}))
        assert result.is_error and "missing required" in result.content

    def test_duplicate_registration_rejected(self):
        registry = ToolRegistry([CalculatorTool()])
        with pytest.raises(ValueError):
            registry.register(CalculatorTool())

    def test_specs_are_json_schema(self):
        spec = ToolRegistry([CalculatorTool()]).specs()[0]
        assert spec.name == "calculator"
        assert spec.parameters["properties"]["expression"]["type"] == "string"


class TestCalculator:
    @pytest.mark.parametrize(
        "expression,expected",
        [("2 + 3 * 4", 14), ("(85 - 60) / 60 * 100", pytest.approx(41.667, rel=1e-3)),
         ("sqrt(16)", 4.0), ("round(3.14159, 2)", 3.14), ("max(3, 9, 1)", 9), ("-5 + 2", -3)],
    )
    def test_arithmetic(self, expression, expected):
        assert CalculatorTool().run(expression=expression)["result"] == expected

    @pytest.mark.parametrize(
        "expression",
        ["__import__('os').system('ls')", "open('/etc/passwd').read()",
         "[].__class__.__base__", "exec('x=1')", "lambda: 1"],
    )
    def test_rejects_code_execution(self, expression):
        """The evaluator walks the AST; it must never reach arbitrary Python."""
        with pytest.raises(ToolError):
            CalculatorTool().run(expression=expression)

    def test_rejects_malformed_input(self):
        with pytest.raises(ToolError):
            CalculatorTool().run(expression="2 +")


class TestSearchTools:
    def test_search_clamps_k(self, retriever):
        assert len(SearchCorpusTool(retriever).run(query="retrieval", k=999)["results"]) <= 20

    def test_search_handles_no_match(self, retriever):
        out = SearchCorpusTool(retriever).run(query="zzzzqqqxyw")
        assert out["results"] == [] or all(r["score"] >= 0 for r in out["results"])

    def test_recall_memory_returns_prior_facts(self, memory):
        memory.remember("Bounded loops keep agent cost predictable.")
        facts = RecallMemoryTool(memory).run(query="bounded agent loops cost")["facts"]
        assert facts and "Bounded loops" in facts[0]["text"]
