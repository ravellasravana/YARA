"""Criteria come from the task; the old constants are only a fallback."""

import copy

import pytest

from yara.agents.decision_agent import DEFAULT_CRITERIA, DecisionAgent

OPTIONS = [
    {"name": "A", "novelty": 0.9, "research_impact": 40, "price": 300, "quality": 0.6},
    {"name": "B", "novelty": 0.3, "research_impact": 95, "price": 100, "quality": 0.9},
    {"name": "C", "novelty": 0.5, "research_impact": 60, "price": 50, "quality": 0.4},
]


def _rank(task):
    result = DecisionAgent(top_n=10).execute({"type": "decision", "data": OPTIONS, **task})
    return [r["option"]["name"] for r in result["recommendations"]], result


def test_task_criteria_drive_the_ranking():
    by_novelty, _ = _rank({"criteria": {"novelty": 1.0}})
    by_impact, _ = _rank({"criteria": {"research_impact": 1.0}})
    assert by_novelty[0] == "A"
    assert by_impact[0] == "B"


def test_no_hidden_bonus_when_criteria_are_given():
    """Old behaviour added novelty/impact on top of whatever the task asked for."""
    names, result = _rank({"criteria": {"quality": 1.0}})
    assert names == ["B", "A", "C"]
    assert set(result["statistics"]["criteria_weights"]) == {"quality"}


def test_falls_back_to_default_criteria():
    _, result = _rank({})
    weights = result["statistics"]["criteria_weights"]
    total = sum(DEFAULT_CRITERIA.values())
    assert weights == pytest.approx({k: v / total for k, v in DEFAULT_CRITERIA.items()})


def test_fallback_uses_all_numeric_fields_for_foreign_options():
    data = [{"name": "x", "speed": 3, "cost": 1}, {"name": "y", "speed": 1, "cost": 1}]
    result = DecisionAgent().execute({"type": "decision", "data": data})
    assert set(result["statistics"]["criteria_weights"]) == {"speed", "cost"}
    assert result["recommendations"][0]["option"]["name"] == "x"


def test_scale_invariance():
    """research_impact is 0-100 and novelty 0-1; equal weights must mean equal pull."""
    data = [
        {"name": "n", "novelty": 1.0, "research_impact": 0},
        {"name": "i", "novelty": 0.0, "research_impact": 100},
    ]
    result = DecisionAgent().execute(
        {"type": "decision", "data": data, "criteria": {"novelty": 1, "research_impact": 1}}
    )
    scores = [r["score"] for r in result["recommendations"]]
    assert scores == pytest.approx([1.0, 1.0])


def test_negative_weight_prefers_lower_values():
    names, _ = _rank({"criteria": {"price": -1.0}})
    assert names == ["C", "B", "A"]


def test_complexity_bonus_comes_from_task():
    data = [
        {"name": "fit", "novelty": 0.5, "implementation_complexity": "low"},
        {"name": "better", "novelty": 0.6, "implementation_complexity": "high"},
    ]
    task = {
        "type": "decision", "data": data, "criteria": {"novelty": 1},
        "user_preferences": {"implementation_complexity": "low"},
    }
    agent = DecisionAgent()
    assert agent.execute(task)["recommendations"][0]["option"]["name"] == "better"
    task["complexity_bonus"] = 1.5
    assert agent.execute(task)["recommendations"][0]["option"]["name"] == "fit"


def test_options_are_not_mutated():
    data = copy.deepcopy(OPTIONS)
    DecisionAgent().execute({"type": "decision", "data": data})
    assert data == OPTIONS


@pytest.mark.parametrize(
    "task",
    [
        {"criteria": ["novelty"]},
        {"criteria": {"novelty": "high"}},
        {"criteria": {"novelty": True}},
        {"criteria": {"novelty": float("nan")}},
        {"complexity_bonus": "big"},
    ],
)
def test_malformed_task_returns_message_not_crash(task):
    _, result = _rank(task)
    assert result["recommendations"] == []
    assert result["message"].startswith("Invalid decision task")


def test_all_zero_weights():
    _, result = _rank({"criteria": {"novelty": 0, "quality": 0}})
    assert result["recommendations"] == []
    assert "zero" in result["message"]


def test_unknown_criterion_is_logged(caplog):
    with caplog.at_level("WARNING", logger="yara.agents.decision"):
        _rank({"criteria": {"novelty": 1, "does_not_exist": 1}})
    assert "does_not_exist" in caplog.text


def test_no_complexity_bonus_without_a_preference():
    """None == None used to award the bonus to every option lacking the field."""
    data = [
        {"name": "tagged", "quality": 0.9, "implementation_complexity": "low"},
        {"name": "untagged", "quality": 0.85},
    ]
    result = DecisionAgent().execute(
        {"type": "decision", "data": data, "criteria": {"quality": 1.0}, "complexity_bonus": 2.0}
    )
    names = [r["option"]["name"] for r in result["recommendations"]]
    assert names == ["tagged", "untagged"]
    assert all(
        "complexity" not in r["reasoning"] for r in result["recommendations"]
    )


def test_cost_only_scores_stay_in_unit_range():
    """Relative scores divide by the best; a negative best put the rest above 1.0."""
    names, result = _rank({"criteria": {"price": -1.0}})
    scores = [r["score"] for r in result["recommendations"]]
    assert names[0] == "C"
    assert scores[0] == pytest.approx(1.0)
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_complexity_bonus_helps_under_cost_criteria():
    """Multiplying a negative score by the bonus used to *penalise* the match."""
    data = [
        {"name": "fit", "price": 100, "implementation_complexity": "low"},
        {"name": "other", "price": 95, "implementation_complexity": "high"},
        {"name": "anchor", "price": 200},
    ]
    result = DecisionAgent().execute(
        {
            "type": "decision",
            "data": data,
            "criteria": {"price": -1.0},
            "complexity_bonus": 3.0,
            "user_preferences": {"implementation_complexity": "low"},
        }
    )
    # Before the fix: fit -1.5 vs other -0.475, so the preferred option lost.
    assert result["recommendations"][0]["option"]["name"] == "fit"
