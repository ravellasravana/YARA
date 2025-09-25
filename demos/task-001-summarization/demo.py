# Text Summarization Demo
from yara.orchestrator import Orchestrator
from yara.agents import SummarizationAgent

# Sample research text
research_text = """
Recent advances in multi-agent systems have demonstrated significant potential
in handling complex research tasks. This study presents a novel approach to
research task orchestration using a modular agent architecture. The methodology
employs weighted decision metrics and automated summarization techniques.

Key findings indicate a 45% improvement in research task completion efficiency
and a 60% increase in accurate decision-making outcomes. The implementation
demonstrates robust handling of multiple research criteria while maintaining
high accuracy in summarization tasks.
"""

# Task configuration
task = {
    "type": "summarization",
    "content": research_text,
    "parameters": {
        "focus": ["methodology", "results"],
        "max_length": 150
    }
}

# Execute task
orchestrator = Orchestrator()
orchestrator.register_agent("summarization", SummarizationAgent())
result = orchestrator.execute_task(task)

print("\nTask 001: Research Text Summarization")
print("="*50)
print("\nInput Text Length:", len(research_text))
print("\nSummarization Results:")
print("-"*30)
for section, content in result["sections"].items():
    print(f"\n{section.title()}:")
    print(content)
print("\nMetadata:")
print(f"Compression Ratio: {result['metadata']['compression_ratio']:.2f}")
print(f"Key Topics: {', '.join(result['metadata']['key_topics'])}")