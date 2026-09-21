"""Weighted multi-criteria ranking. No model call - same input, same output."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

import numpy as np

logger = logging.getLogger("yara.agents.decision")

# Fallbacks for tasks that do not say what matters. These were the hand-tuned
# constants the agent used to apply to every task; they now only apply when
# the task is silent, so a caller who does pass criteria gets exactly the
# ranking they asked for, with no hidden bonus on top.
DEFAULT_CRITERIA: dict[str, float] = {"novelty": 0.5, "research_impact": 0.7}
DEFAULT_COMPLEXITY_BONUS = 1.1


class DecisionAgent:
    def __init__(self, top_n: int = 3):
        self.logger = logger
        self.top_n = top_n

    def execute(self, task: dict[str, Any]) -> dict[str, Any]:
        """Rank ``task["data"]`` options against the task's own criteria.

        Recognised task keys:

        - ``criteria``: ``{field: weight}``. A negative weight marks a cost
          (lower is better, e.g. ``{"price": -0.3}``). Omitted means
          :data:`DEFAULT_CRITERIA`, or equal weights over every numeric field
          if the options carry none of the default fields.
        - ``complexity_bonus``: multiplier for options whose
          ``implementation_complexity`` matches the user's preference.
        - ``user_preferences``: hard constraints and soft preferences.
        """
        if task.get("type") != "decision":
            return {"type": "recommendation", "recommendation": "N/A (not applicable for this task)"}

        options = _extract_options(task.get("data", {}))
        if not options:
            return _empty("No options available for decision making")

        try:
            criteria = _resolve_criteria(task.get("criteria"), options)
            bonus = _as_weight(
                task.get("complexity_bonus", DEFAULT_COMPLEXITY_BONUS), "complexity_bonus"
            )
        except ValueError as exc:
            return _empty(f"Invalid decision task: {exc}")

        total = sum(abs(w) for w in criteria.values())
        if total == 0:
            return _empty("All criteria weights are zero; nothing to rank on")
        # Divide by the sum of |w| rather than sum(w) so a cost criterion can't
        # flip the sign of every other weight.
        weights = {k: w / total for k, w in criteria.items()}
        missing = [k for k in weights if not any(_is_number(o.get(k)) for o in options)]
        if missing:
            self.logger.warning("criteria not present on any option: %s", ", ".join(missing))

        prefs = task.get("user_preferences") or {}
        scales = _scales(options, weights)

        scored = []
        for o in options:
            normalised = _normalise(o, scales)
            s = self._score(o, normalised, weights, prefs, bonus)
            if s is not None:
                scored.append((o, normalised, s))

        if not scored:
            return _empty("No suitable options found after applying criteria")

        scored.sort(key=lambda item: -item[2])
        best = scored[0][2] or 1.0
        self.logger.info("ranked %d of %d options", len(scored), len(options))

        return {
            "type": "recommendation",
            "recommendations": [
                {"option": o, "score": float(s / best), "reasoning": self._why(o, n, prefs)}
                for o, n, s in scored[: self.top_n]
            ],
            "statistics": _stats([s for _, _, s in scored], len(options), weights),
        }

    def _score(
        self,
        option: dict[str, Any],
        normalised: dict[str, float],
        weights: dict[str, float],
        prefs: dict[str, Any],
        bonus: float,
    ) -> float | None:
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

        for field, w in weights.items():
            score += normalised.get(field, 0.0) * w

        if option.get("implementation_complexity") == prefs.get("implementation_complexity"):
            score *= bonus
        return score

    def _why(
        self, option: dict[str, Any], normalised: dict[str, float], prefs: dict[str, Any]
    ) -> str:
        reasons = []
        if option.get("novelty", 0) > 0.7:
            reasons.append("High novelty factor")
        if normalised.get("research_impact", 0) > 0.7:
            reasons.append("Strong research impact potential")
        if option.get("implementation_complexity") == prefs.get("implementation_complexity"):
            reasons.append("Matches preferred implementation complexity")
        matched = set(option.get("features", [])) & set(prefs.get("required_features", []))
        if matched:
            reasons.append(f"Contains required features: {', '.join(sorted(matched))}")
        return " | ".join(reasons) or "Based on overall score analysis"


def _empty(message: str) -> dict[str, Any]:
    return {"type": "recommendation", "message": message, "recommendations": []}


def _is_number(v: Any) -> bool:
    # bool is an int subclass; True/False are flags, not magnitudes.
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _as_weight(v: Any, name: str) -> float:
    if not _is_number(v) or not np.isfinite(v):
        raise ValueError(f"{name} must be a finite number, got {v!r}")
    return float(v)


def _resolve_criteria(raw: Any, options: list[dict[str, Any]]) -> dict[str, float]:
    """Task criteria if given, else the defaults the options can actually use."""
    if raw:
        if not isinstance(raw, Mapping):
            raise ValueError("criteria must be a mapping of field name to weight")
        return {str(k): _as_weight(v, f"criteria[{k!r}]") for k, v in raw.items()}

    present = {
        k: w for k, w in DEFAULT_CRITERIA.items() if any(_is_number(o.get(k)) for o in options)
    }
    if present:
        return present
    # Options from some other domain (price/quality, say): with no guidance,
    # every numeric field counts equally.
    return {k: 1.0 for k, v in options[0].items() if _is_number(v)}


def _scales(options: list[dict[str, Any]], weights: dict[str, float]) -> dict[str, float]:
    """Largest magnitude per criterion, so 0-100 and 0-1 fields weigh the same.

    Taken over all options, not just those that survive the constraints, so a
    score does not shift when an unrelated option is filtered out.
    """
    scales = {}
    for field in weights:
        values = [abs(float(o[field])) for o in options if _is_number(o.get(field))]
        scales[field] = max(values, default=0.0) or 1.0
    return scales


def _normalise(option: dict[str, Any], scales: dict[str, float]) -> dict[str, float]:
    # Kept apart from the option so callers get their data back unmodified.
    return {f: float(option[f]) / s for f, s in scales.items() if _is_number(option.get(f))}


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
