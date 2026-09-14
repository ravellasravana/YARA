"""Agent parsing and fallbacks, against a scripted provider so failure modes
can be forced."""

from yara.agents.research import AnalysisAgent, CriticAgent, PlannerAgent, SynthesisAgent
from yara.llm.base import LLMResponse, Usage


class Stub:
    name = "stub"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append(list(messages))
        text = self.responses.pop(0) if self.responses else "{}"
        return LLMResponse(text=text, usage=Usage(10, 5))


class TestPlanner:
    def test_parses_subquestions(self):
        out = PlannerAgent(
            Stub('{"subquestions":[{"id":"sq1","question":"What is X?","angle":"definition"}]}')
        ).run(question="What is X?").output
        assert len(out) == 1 and out[0].id == "sq1"
        assert out[0].search_query == "What is X?"

    def test_falls_back_to_a_single_node_plan(self):
        r = PlannerAgent(Stub("not json at all")).run(question="What is X?")
        assert not r.ok
        assert len(r.output) == 1 and r.output[0].question == "What is X?"

    def test_role_marker_is_in_the_system_prompt(self):
        stub = Stub('{"subquestions":[]}')
        PlannerAgent(stub).run(question="q")
        assert "YARA_ROLE: planner" in stub.calls[0][0].content


class TestAnalyst:
    def test_normalises_a_string_citation(self):
        out = AnalysisAgent(
            Stub('{"findings":[{"claim":"C","citations":"c1","confidence":0.9}]}')
        ).run(subquestion="q", evidence=[]).output
        assert out[0].citations == ["c1"]

    def test_drops_findings_with_no_claim(self):
        out = AnalysisAgent(Stub('{"findings":[{"citations":["c1"]}, {"claim":"ok"}]}')).run(
            subquestion="q", evidence=[]
        ).output
        assert [f.claim for f in out] == ["ok"]

    def test_falls_back_to_empty(self):
        assert AnalysisAgent(Stub("<<<")).run(subquestion="q", evidence=[]).output == []


class TestSynthesiser:
    def test_falls_back_to_findings_as_sections(self):
        from yara.briefs.schema import Finding

        r = SynthesisAgent(Stub("garbage")).run(
            question="q", findings=[Finding(claim="A", citations=["c1"])]
        )
        assert not r.ok
        assert r.output["sections"][0]["body"] == "A"
        assert r.output["sections"][0]["citations"] == ["c1"]


class TestCritic:
    def test_parses_a_verdict(self):
        out = CriticAgent(Stub('{"verdict":"pass","coverage":1.0,"issues":[]}')).run(
            claims=[], evidence_ids=[]
        ).output
        assert out.passed

    def test_fails_closed(self):
        """Unparseable critique must never read as approval. This is the
        default a later refactor would most want to flip for convenience."""
        r = CriticAgent(Stub("<<garbage>>")).run(claims=[], evidence_ids=[])
        assert not r.ok
        assert r.output.verdict == "revise"


class TestToolLoop:
    def test_stops_when_model_answers(self):
        from yara.llm.base import ToolCall
        from yara.tools.base import Tool, ToolRegistry

        class Echo(Tool):
            name = "echo"
            description = "echo"
            parameters = {"type": "object", "properties": {"s": {"type": "string"}}}

            def run(self, s=""):
                return s

        class ToolThenAnswer:
            name = "stub"

            def __init__(self):
                self.n = 0

            def complete(self, messages, tools=None, **_):
                self.n += 1
                if self.n == 1:
                    return LLMResponse(text="", tool_calls=[ToolCall("c1", "echo", {"s": "hi"})])
                return LLMResponse(text='{"subquestions":[]}')

        agent = PlannerAgent(ToolThenAnswer(), tools=ToolRegistry([Echo()]))
        r = agent.run(question="q", use_tools=True)
        assert r.tool_calls == 1
        assert r.iterations == 2

    def test_budget_is_enforced(self):
        from yara.llm.base import ToolCall
        from yara.tools.base import Tool, ToolRegistry

        class Echo(Tool):
            name = "echo"
            description = "echo"
            parameters = {"type": "object", "properties": {}}

            def run(self):
                return "x"

        class AlwaysTools:
            name = "stub"

            def complete(self, messages, tools=None, **_):
                if tools is None:
                    return LLMResponse(text='{"subquestions":[]}')
                return LLMResponse(text="", tool_calls=[ToolCall("c", "echo", {})])

        agent = PlannerAgent(AlwaysTools(), tools=ToolRegistry([Echo()]), max_iterations=3)
        r = agent.run(question="q", use_tools=True)
        assert r.tool_calls == 3
        assert r.ok
