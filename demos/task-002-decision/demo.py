# Decision Making Demo
from yara.orchestrator import Orchestrator
from yara.agents import DecisionAgent

# Research project options
research_options = [
    {
        "name": "Machine Learning Approach",
        "novelty": 0.8,
        "research_impact": 85,
        "implementation_complexity": "medium",
        "features": ["core", "innovative", "scalable"],
        "price": 100
    },
    {
        "name": "Statistical Analysis",
        "novelty": 0.5,
        "research_impact": 65,
        "implementation_complexity": "low",
        "features": ["core", "established"],
        "price": 50
    },
    {
        "name": "Neural Architecture",
        "novelty": 0.9,
        "research_impact": 90,
        "implementation_complexity": "high",
        "features": ["experimental", "innovative"],
        "price": 200
    }
]

# Task configuration
task = {
    "type": "decision",
    "data": research_options,
    "user_preferences": {
        "implementation_complexity": "medium",
        "required_features": ["core"],
        "max_price": 150
    },
    "criteria": {
        "novelty": 0.4,
        "research_impact": 0.6
    }
}

# Execute task
orchestrator = Orchestrator()
orchestrator.register_agent("decision", DecisionAgent())
result = orchestrator.execute_task(task)

print("\nTask 002: Research Decision Making")
print("="*50)
print("\nAnalyzing Research Options:")
print("-"*30)

for rec in result["recommendations"]:
    print(f"\nOption: {rec['option']['name']}")
    print(f"Score: {rec['score']:.2f}")
    print(f"Reasoning: {rec['reasoning']}")
    print("\nMetrics:")
    print(f"- Novelty: {rec['option']['novelty']:.2f}")
    print(f"- Research Impact: {rec['option']['research_impact']}/100")
    print(f"- Complexity: {rec['option']['implementation_complexity']}")
    print(f"- Features: {', '.join(rec['option']['features'])}")
    print(f"- Cost: ${rec['option']['price']}")