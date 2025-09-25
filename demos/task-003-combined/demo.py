# Combined Analysis Demo
from yara.orchestrator import Orchestrator
from yara.agents import DecisionAgent, SummarizationAgent

# Research methodology text
methodology_text = """
This study employs three distinct approaches to natural language processing:
1. Traditional statistical methods with established accuracy (65%)
2. Modern transformer-based architecture with high novelty (90%)
3. Hybrid approach combining both methods with balanced performance (80%)

Each approach was evaluated on a standard dataset using accuracy, 
innovation potential, and implementation complexity as key metrics.
"""

# First step: Summarize methodology
summarize_task = {
    "type": "summarization",
    "content": methodology_text,
    "parameters": {
        "focus": "methodology",
        "max_length": 100
    }
}

# Second step: Decision making based on summary
decision_task = {
    "type": "decision",
    "data": [
        {
            "name": "Statistical Approach",
            "novelty": 0.4,
            "research_impact": 65,
            "implementation_complexity": "low",
            "features": ["established", "reliable"],
            "price": 50
        },
        {
            "name": "Transformer Approach",
            "novelty": 0.9,
            "research_impact": 90,
            "implementation_complexity": "high",
            "features": ["innovative", "scalable"],
            "price": 200
        },
        {
            "name": "Hybrid Approach",
            "novelty": 0.7,
            "research_impact": 80,
            "implementation_complexity": "medium",
            "features": ["balanced", "reliable", "scalable"],
            "price": 150
        }
    ],
    "criteria": {
        "novelty": 0.4,
        "research_impact": 0.6
    }
}

# Execute combined analysis
orchestrator = Orchestrator()
orchestrator.register_agent("summarization", SummarizationAgent())
orchestrator.register_agent("decision", DecisionAgent())

print("\nTask 003: Combined Research Analysis")
print("="*50)

# Step 1: Summarization
print("\nStep 1: Methodology Summary")
print("-"*30)
summary_result = orchestrator.execute_task(summarize_task)
print(summary_result["sections"]["methodology"])

# Step 2: Decision Making
print("\nStep 2: Approach Selection")
print("-"*30)
decision_result = orchestrator.execute_task(decision_task)

for rec in decision_result["recommendations"]:
    print(f"\nRecommended: {rec['option']['name']}")
    print(f"Confidence Score: {rec['score']:.2f}")
    print(f"Reasoning: {rec['reasoning']}")

# Final Analysis
print("\nFinal Analysis")
print("-"*30)
print("Based on methodology assessment and approach evaluation:")
top_rec = decision_result["recommendations"][0]
print(f"- Selected Approach: {top_rec['option']['name']}")
print(f"- Confidence Level: {top_rec['score']:.2f}")
print(f"- Key Factors: {top_rec['reasoning']}")