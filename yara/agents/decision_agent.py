from typing import Dict, Any, List, Optional, Tuple
import logging
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import json

class DecisionAgent:
    def __init__(self):
        self.logger = logging.getLogger("YARA.DecisionAgent")
        self.scaler = MinMaxScaler()

    def _calculate_option_score(self, 
                              option: Dict[str, Any], 
                              normalized_criteria: Dict[str, float],
                              user_prefs: Dict[str, Any]) -> Optional[float]:
        """Calculate score for a single option"""
        # Check required features
        if "required_features" in user_prefs:
            option_features = set(option.get("features", []))
            required_features = set(user_prefs["required_features"])
            if not required_features.issubset(option_features):
                return None

        # Check constraints
        if "max_price" in user_prefs and option.get("price", float("inf")) > user_prefs["max_price"]:
            return None
        if "min_quality" in user_prefs and option.get("quality", 0) < user_prefs["min_quality"]:
            return None

        # Base score
        score = 0.0

        # Add availability preference
        if "preferred_availability" in user_prefs and user_prefs["preferred_availability"] == option.get("availability"):
            score += 1.0

        # Add novelty score (50% weight)
        novelty_score = option.get("novelty", 0)
        score += novelty_score * 0.5

        # Add research impact score (70% weight)
        research_score = option.get("research_impact_score", 0)
        score += research_score * 0.7

        # Add weighted criteria scores
        for criterion, weight in normalized_criteria.items():
            if criterion in option and isinstance(option[criterion], (int, float)):
                value = float(option[criterion])
                # Normalize value against maximum in options if needed
                score += value * weight

        # Add complexity match bonus (10%)
        if option.get("implementation_complexity") == user_prefs.get("implementation_complexity"):
            score *= 1.1

        return score

    def _generate_reasoning(self, 
                          option: Dict[str, Any], 
                          score: float,
                          user_prefs: Dict[str, Any]) -> str:
        """Generate reasoning for recommendation"""
        reasons = []
        
        if option.get("novelty", 0) > 0.7:
            reasons.append("High novelty factor")
        
        if option.get("research_impact_score", 0) > 0.7:
            reasons.append("Strong research impact potential")
        
        if option.get("implementation_complexity") == user_prefs.get("implementation_complexity"):
            reasons.append("Matches preferred implementation complexity")
        
        if option.get("features", []):
            matched_features = set(option["features"]).intersection(
                set(user_prefs.get("required_features", [])))
            if matched_features:
                reasons.append(f"Contains required features: {', '.join(matched_features)}")

        if not reasons:
            reasons.append("Based on overall score analysis")

        return " | ".join(reasons)

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute decision tasks with research metrics"""
        # Validate task type
        if task.get("type") != "decision":
            return {
                "type": "recommendation",
                "recommendation": "N/A (not applicable for this task)"
            }

        # Extract data, user preferences, and criteria
        data = task.get("data", {})
        user_prefs = task.get("user_preferences", {})
        task_criteria = task.get("criteria", {})

        # Parse data if it's a string
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                data = {}

        # Extract options from data
        options = []
        if isinstance(data, list):
            options = data
        elif isinstance(data, dict):
            content = data.get("content", data)
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    content = {}
            options = content.get("options", []) if isinstance(content, dict) else content

        if not options:
            return {
                "type": "recommendation",
                "message": "No options available for decision making",
                "recommendations": []
            }

        # Normalize research impact scores
        max_impact = max((opt.get("research_impact", 0) for opt in options), default=100)
        for opt in options:
            if "research_impact" in opt:
                opt["research_impact_score"] = opt["research_impact"] / max_impact

        # Prepare scoring criteria
        numeric_fields = [k for k, v in (options[0] if options else {}).items() 
                        if isinstance(v, (int, float))]
        criteria = task_criteria if task_criteria else {field: 1.0 for field in numeric_fields}
        
        # Normalize weights
        total_weight = sum(criteria.values())
        if total_weight == 0:
            total_weight = 1
        normalized_criteria = {k: v/total_weight for k, v in criteria.items()}

        # Score options
        scored_options = []
        for option in options:
            score = self._calculate_option_score(option, normalized_criteria, user_prefs)
            if score is not None:  # None indicates option should be filtered out
                scored_options.append((option, score))

        if not scored_options:
            return {
                "type": "recommendation",
                "message": "No suitable options found after applying criteria",
                "recommendations": []
            }

        # Sort by score and normalize scores
        ranked_options = sorted(scored_options, key=lambda x: x[1], reverse=True)
        max_score = max(s[1] for s in scored_options)

        # Generate recommendations with reasoning
        recommendations = []
        for opt, score in ranked_options[:3]:  # Top 3 recommendations
            recommendations.append({
                "option": opt,
                "score": float(score / max_score),  # Normalize to [0,1]
                "reasoning": self._generate_reasoning(opt, score, user_prefs)
            })

        return {
            "type": "recommendation",
            "recommendations": recommendations
        }
        # Validate task type
        if task.get("type") != "decision":
            return {
                "type": "recommendation",
                "recommendation": "N/A (not applicable for this task)"
            }

        # Extract data, user preferences, and criteria
        data = task.get("data", {})
        user_prefs = task.get("user_preferences", {})
        task_criteria = task.get("criteria", {})

        # Parse data if it's a string
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                data = {}

        # Extract options from data
        options = []
        if isinstance(data, list):
            options = data
        elif isinstance(data, dict):
            options = data.get("options", [])
            if isinstance(options, str):
                try:
                    options = json.loads(options)
                except json.JSONDecodeError:
                    options = []

        if not options:
            return {
                "type": "recommendation",
                "message": "No options available for decision making",
                "recommendations": []
            }
        task_type = task.get("type", "unknown")
        if task_type != "decision":
            return {
                "type": "recommendation",
                "recommendation": "N/A (not applicable for this task)"
            }

        # Get data from either task data or previous retrieval result
        data = task.get("data", {})
        if isinstance(data, dict) and "content" in data:
            data = data["content"]
            
        if isinstance(data, str):
            try:
                import json
                data = json.loads(data)
            except:
                data = {}
                
            # Handle both direct options and nested options
        options = []
        if isinstance(data, list):
            options = data
        elif isinstance(data, dict):
            options = data.get("options", [])
        
        # Get user preferences and criteria
        user_prefs = task.get("user_preferences", {})
        task_criteria = task.get("criteria", {})
        
        # Convert research_impact to normalized score
        max_impact = max((opt.get("research_impact", 0) for opt in options), default=100)
        for opt in options:
            if "research_impact" in opt:
                opt["research_impact_score"] = opt["research_impact"] / max_impact
        
        self.logger.info(f"Making recommendation with {len(options)} options")

        # Normalize the criteria
        criteria = {}
        for key, value in task_criteria.items():
            if isinstance(value, (int, float)):
                criteria[key] = float(value)
        
        if not options:
            return {
                "type": "recommendation",
                "message": "No options available for decision making",
                "recommendations": []
            }

        # Combine user preferences with criteria
        combined_criteria = {}
        if "max_price" in user_prefs:
            combined_criteria["price"] = lambda x: 1.0 if x <= user_prefs["max_price"] else 0.0
        if "min_quality" in user_prefs:
            combined_criteria["quality"] = lambda x: 1.0 if x >= user_prefs["min_quality"] else 0.0
        
        # Add weighted criteria from task
        for key, weight in criteria.items():
            if key in combined_criteria:
                continue
            combined_criteria[key] = lambda x, w=weight: float(x) * float(w)

        # Score each option
        scored_options = []
        for option in options:
            base_score = 0.0
            valid_option = True
            
            # Check required features
            if "required_features" in user_prefs:
                option_features = set(option.get("features", []))
                required_features = set(user_prefs["required_features"])
                if not required_features.issubset(option_features):
                    continue

            # Check constraints
            if "max_price" in user_prefs and option.get("price", float("inf")) > user_prefs["max_price"]:
                continue
            if "min_quality" in user_prefs and option.get("quality", 0) < user_prefs["min_quality"]:
                continue
            
            # Add base score for availability preference
            if "preferred_availability" in user_prefs and user_prefs["preferred_availability"] == option.get("availability"):
                base_score += 1.0
                
            # Add novelty score
            novelty_score = option.get("novelty", 0)
            if novelty_score > 0:
                base_score += novelty_score * 0.5  # Weight novelty at 50%
                
            # Add research impact score
            research_score = option.get("research_impact_score", 0)
            if research_score > 0:
                base_score += research_score * 0.7  # Weight research impact at 70%

            # Calculate weighted score based on research criteria
            total_weight = 0
            for key, weight in criteria.items():
                if key == "research_impact" and "research_impact_score" in option:
                    base_score += weight * option["research_impact_score"]
                    total_weight += weight
                elif key == "novelty" and key in option:
                    base_score += weight * option[key]
                    total_weight += weight
                elif key in option and isinstance(option[key], (int, float)):
                    normalized_value = option[key] / max((opt.get(key, 1) for opt in options), default=1)
                    base_score += weight * normalized_value
                    total_weight += weight

            # Normalize the score
            if total_weight > 0:
                base_score = base_score / total_weight

            if valid_option:
                # Add implementation complexity bonus/penalty
                if option.get("implementation_complexity") == user_prefs.get("implementation_complexity"):
                    base_score *= 1.1  # 10% bonus for matching complexity preference
                
                scored_options.append({
                    "option": option,
                    "score": base_score
                })

        # Sort by score
        scored_options.sort(key=lambda x: x["score"], reverse=True)
        
        # Calculate statistical metrics
        scores = [opt["score"] for opt in scored_options]
        stats = {
            "mean_score": float(np.mean(scores)) if scores else 0,
            "std_dev": float(np.std(scores)) if scores else 0,
            "median": float(np.median(scores)) if scores else 0,
            "min_score": float(np.min(scores)) if scores else 0,
            "max_score": float(np.max(scores)) if scores else 0,
            "total_options": len(options),
            "valid_options": len(scored_options),
            "criteria_weights": criteria,
            "timestamp": task.get("timestamp", ""),
        }

        # Add percentile rankings
        if scores:
            percentiles = [25, 50, 75, 90]
            stats["percentiles"] = {
                f"p{p}": float(np.percentile(scores, p))
                for p in percentiles
            }
        
        return {
            "type": "recommendation",
            "recommendations": recommendations
        }

    def _make_classification(self, options: List[Dict[str, Any]], criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Classify options based on criteria"""
        threshold = criteria.get("threshold", 0.5)
        features = criteria.get("features", [])

        if not options or not features:
            raise ValueError("Options and features are required for classification")

        classifications = []
        for option in options:
            # Extract feature values
            feature_values = []
            for feature in features:
                value = option.get(feature, 0)
                feature_values.append(float(value))

            # Normalize features
            normalized_values = self.scaler.fit_transform([feature_values])[0]
            
            # Simple threshold-based classification
            score = np.mean(normalized_values)
            classification = "accept" if score >= threshold else "reject"

            classifications.append({
                "option": option,
                "classification": classification,
                "confidence": float(abs(score - threshold))
            })

        return {
            "type": "classification",
            "classifications": classifications,
            "threshold": threshold
        }
