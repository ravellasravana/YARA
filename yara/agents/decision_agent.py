"""Weighted multi-criteria ranking. No model call - same input, same output."""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

logger = logging.getLogger("yara.agents.decision")

# Tuned by hand against the option sets in tests/test_decision_agent.py.
# TODO: these should come from the task, not be baked in here.
NOVELTY_WEIGHT = 0.5
IMPACT_WEIGHT = 0.7
COMPLEXITY_BONUS = 1.1


class DecisionAgent:
    def __init__(self, top_n: int = 3):
        self.logger = logger
        self.top_n = top_n

    def execute(self, task: dict[str, Any]) -> dict[str, Any]:
        if task.get("type") != "decision":
            return {"type": "recommendation", "recommendation": "N/A (not applicable for this task)"}

        options = _extract_options(task.get("data", {}))
        if not options:
            return _empty("No options available for decision making")

        prefs = task.get("user_preferences") or {}

        # research_impact arrives on a 0-100 scale, novelty on 0-1. Rescale so
        # the weights below mean the same thing for both.
        top_impact = max((o.get("research_impact", 0) for o in options), default=0) or 100
        for o in options:
            if "research_impact" in o:
                o["research_impact_score"] = o["research_impact"] / top_impact

        criteria = task.get("criteria") or _numeric_fields(options[0])
        total = sum(criteria.values()) or 1.0
        weights = {k: v / total for k, v in criteria.items()}

        scored = []
        for o in options:
            s = self._score(o, weights, prefs)
            if s is not None:
                scored.append((o, s))

        if not scored:
            return _empty("No suitable options found after applying criteria")

        scored.sort(key=lambda pair: -pair[1])
        best = scored[0][1] or 1.0
        self.logger.info("ranked %d of %d options", len(scored), len(options))

        return {
            "type": "recommendation",
            "recommendations": [
                {"option": o, "score": float(s / best), "reasoning": self._why(o, prefs)}
                for o, s in scored[: self.top_n]
            ],
            "statistics": _stats([s for _, s in scored], len(options), weights),
        }

    def _score(self, option, weights, prefs) -> float | None:
        """None means the option was filtered out by a hard constraint."""
        required = set(prefs.get("required_features", []))
        if required and not required.issubset(set(option.get("features", []))):
            return None
        if "max_price" in prefs and option.get("price", float("inf")) > prefs["max_price"]:
            return None
        if "min_quality" in prefs and option.get("quality", 0) < prefs["min_quality"]:
            return None

        score = 0.0
        if "preferred_availability" in prefs:
            if prefs["preferred_availability"] == option.get("availability"):
                score += 1.0

        score += option.get("novelty", 0) * NOVELTY_WEIGHT
        score += option.get("research_impact_score", 0) * IMPACT_WEIGHT

        for field, w in weights.items():
            v = option.get(field)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                score += float(v) * w

        if option.get("implementation_complexity") == prefs.get("implementation_complexity"):
            score *= COMPLEXITY_BONUS
        return score

    def _why(self, option, prefs) -> str:
        reasons = []
        if option.get("novelty", 0) > 0.7:
            reasons.append("High novelty factor")
        if option.get("research_impact_score", 0) > 0.7:
            reasons.append("Strong research impact potential")
        if option.get("implementation_complexity") == prefs.get("implementation_complexity"):
            reasons.append("Matches preferred implementation complexity")
        matched = set(option.get("features", [])) & set(prefs.get("required_features", []))
        if matched:
            reasons.append(f"Contains required features: {', '.join(sorted(matched))}")
        return " | ".join(reasons) or "Based on overall score analysis"


def _empty(message: str) -> dict[str, Any]:
    return {"type": "recommendation", "message": message, "recommendations": []}


def _numeric_fields(option: dict[str, Any]) -> dict[str, float]:
    return {
        k: 1.0
        for k, v in option.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }


def _extract_options(data: Any) -> list[dict[str, Any]]:
    """Callers pass a list, a {"options": [...]}, a {"content": ...} wrapper, or a JSON string."""
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return []
    if isinstance(data, list):
        return [o for o in data if isinstance(o, dict)]
    if isinstance(data, dict):
        if "content" in data:
            return _extract_options(data["content"])
        return _extract_options(data.get("options", []))
    return []


def _stats(scores: list[float], total_options: int, weights: dict[str, float]) -> dict[str, Any]:
    if not scores:
        return {"total_options": total_options, "valid_options": 0}
    a = np.asarray(scores, dtype=float)
    return {
        "mean_score": float(a.mean()),
        "std_dev": float(a.std()),
        "median": float(np.median(a)),
        "min_score": float(a.min()),
        "max_score": float(a.max()),
        "total_options": total_options,
        "valid_options": len(scores),
        "criteria_weights": weights,
        "percentiles": {f"p{p}": float(np.percentile(a, p)) for p in (25, 50, 75, 90)},
    }
