import pytest

from yara.llm.base import (
    Message,
    ToolCall,
    ToolResult,
    ToolSpec,
    parse_json_loose,
    split_system,
)
from yara.llm.echo import EchoProvider

SPEC = ToolSpec(
    name="search",
    description="search the corpus",
    parameters={
        "type": "object",
        "properties": {"q": {"type": "string", "default": "x"}},
        "required": ["q"],
        "additionalProperties": False,
    },
)


class TestJSONParsing:
    @pytest.mark.parametrize(
        "raw",
        [
            '{"a": 1}',
            '```json\n{"a": 1}\n```',
            '```\n{"a": 1}\n```',
            'Here you go:\n{"a": 1}\nHope that helps.',
        ],
    )
    def test_digs_json_out_of_prose_and_fences(self, raw):
        assert parse_json_loose(raw) == {"a": 1}

    def test_handles_top_level_arrays(self):
        assert parse_json_loose("prefix [1, 2, 3] suffix") == [1, 2, 3]

    def test_raises_when_there_is_nothing(self):
        with pytest.raises(ValueError):
            parse_json_loose("no json here at all")


class TestSystemSplit:
    def test_joins_multiple_system_turns(self):
        system, rest = split_system(
            [Message.system("a"), Message.user("q"), Message.system("b")]
        )
        assert system == "a\n\nb"
        assert [m.role for m in rest] == ["user"]


class TestToolSpecTranslation:
    def test_anthropic_uses_input_schema(self):
        assert SPEC.to_anthropic()["input_schema"]["required"] == ["q"]

    def test_openai_wraps_in_a_function_object(self):
        payload = SPEC.to_openai()
        assert payload["type"] == "function"
        assert payload["function"]["name"] == "search"

    def test_gemini_strips_keywords_it_rejects(self):
        """A stray additionalProperties or default is a 400 from Gemini."""
        params = SPEC.to_gemini()["parameters"]
        assert "additionalProperties" not in params
        assert "default" not in params["properties"]["q"]
        assert params["properties"]["q"]["type"] == "string"


class TestMessages:
    def test_tool_message_carries_results(self):
        m = Message.tool([ToolResult("c1", "search", "found")])
        assert m.role == "tool"
        assert m.tool_results[0].call_id == "c1"

    def test_assistant_can_hold_tool_calls(self):
        m = Message.assistant("thinking", [ToolCall("c1", "search", {"q": "x"})])
        assert m.tool_calls[0].name == "search"


class TestEchoProvider:
    def test_is_deterministic(self):
        p = EchoProvider()
        msgs = [Message.system("YARA_ROLE: planner"), Message.user("<QUESTION>What is X?</QUESTION>")]
        assert p.complete(msgs).text == p.complete(msgs).text

    def test_planner_returns_subquestions(self):
        out = EchoProvider().complete([
            Message.system("YARA_ROLE: planner"),
            Message.user("<QUESTION>How does hybrid retrieval work?</QUESTION>\n<MAX_SUBQUESTIONS>3</MAX_SUBQUESTIONS>"),
        ]).json()
        assert len(out["subquestions"]) == 3
        assert all(sq["question"] for sq in out["subquestions"])

    def test_analysis_only_cites_chunks_it_was_given(self):
        """Extractive, so it cannot invent a chunk id - that's the point of it."""
        out = EchoProvider().complete([
            Message.system("YARA_ROLE: analysis"),
            Message.user(
                "<SUBQUESTION>What bounds answer quality?</SUBQUESTION>\n"
                "<EVIDENCE>\n[c1] Retrieval quality bounds answer quality directly. "
                "A generator fed poor evidence produces confident wrong answers.\n</EVIDENCE>"
            ),
        ]).json()
        assert out["findings"]
        assert all(f["citations"] == ["c1"] for f in out["findings"])

    def test_analysis_claims_come_verbatim_from_evidence(self):
        passage = "Recall at k sets a ceiling on what the generator can ground an answer in."
        out = EchoProvider().complete([
            Message.system("YARA_ROLE: analysis"),
            Message.user(f"<SUBQUESTION>What is recall at k?</SUBQUESTION>\n<EVIDENCE>\n[c1] {passage}\n</EVIDENCE>"),
        ]).json()
        assert out["findings"][0]["claim"] in passage

    def test_critic_flags_an_uncited_claim(self):
        out = EchoProvider().complete([
            Message.system("YARA_ROLE: critic"),
            Message.user('<CLAIMS>[{"claim": "Something asserted", "citations": []}]</CLAIMS>\n<EVIDENCE_IDS>c1</EVIDENCE_IDS>'),
        ]).json()
        assert out["verdict"] == "revise"
        assert out["issues"][0]["issue"] == "uncited"

    def test_critic_flags_a_citation_that_does_not_resolve(self):
        out = EchoProvider().complete([
            Message.system("YARA_ROLE: critic"),
            Message.user('<CLAIMS>[{"claim": "X", "citations": ["made_up"]}]</CLAIMS>\n<EVIDENCE_IDS>c1, c2</EVIDENCE_IDS>'),
        ]).json()
        assert out["verdict"] == "revise"

    def test_critic_passes_a_grounded_claim(self):
        out = EchoProvider().complete([
            Message.system("YARA_ROLE: critic"),
            Message.user('<CLAIMS>[{"claim": "X", "citations": ["c1"]}]</CLAIMS>\n<EVIDENCE_IDS>c1, c2</EVIDENCE_IDS>'),
        ]).json()
        assert out["verdict"] == "pass"

    def test_reports_token_usage(self):
        r = EchoProvider().complete([Message.user("some text here")])
        assert r.usage.input_tokens > 0
