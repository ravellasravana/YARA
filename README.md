# YARA: Multi-Agent Research Framework

## Overview
YARA (Yet Another Research Assistant) is a sophisticated multi-agent framework designed for academic research tasks. It implements a modular architecture with advanced task orchestration, focusing on research-driven decision making and automated analysis.

## Core Components

### 1. Multi-Agent Architecture
- **Orchestrator**: Coordinates agent interactions and task flow
- **Decision Agent**: Research-focused option evaluation
- **Summarization Agent**: Advanced text analysis and synthesis
- **Memory Handler**: Context management and persistence

### 2. Research Features
- Multi-criteria decision analysis
- Automated research summarization
- Weighted evaluation metrics
- Innovation assessment scoring

## Key Features

1. **Multi-criteria Evaluation**
   - Innovation metrics (novelty and research impact)
   - Implementation characteristics
   - Resource requirements
   - Feature compatibility

2. **Intelligent Scoring**
   - Weighted criteria analysis
   - Normalized scoring across dimensions
   - Constraint satisfaction
   - Preference matching

3. **Research-Oriented Output**
   - Detailed analysis of each option
   - Quantified confidence levels
   - Clear decision reasoning
   - Validation metrics

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/yara-decision-agent.git

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
from yara.agents.decision_agent import DecisionAgent

# Initialize the agent
agent = DecisionAgent()

# Prepare decision task
task = {
    "type": "decision",
    "data": [
        {
            "name": "Option A",
            "novelty": 0.8,
            "research_impact": 85,
            "implementation_complexity": "medium",
            "features": ["core", "innovative"],
            "price": 100
        },
        # Additional options...
    ],
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

# Get recommendations
result = agent.execute(task)
```

## Research Implementation

The decision agent employs a sophisticated algorithm that:

1. **Innovation Assessment**
   - Evaluates novelty scores (0-1 scale)
   - Analyzes research impact (0-100 scale)
   - Weights contributions based on research priorities

2. **Feasibility Analysis**
   - Validates implementation complexity
   - Checks feature requirements
   - Ensures resource constraints

3. **Optimization**
   - Normalizes scores across dimensions
   - Applies weighted criteria
   - Ranks options by composite scores

## Testing

The system includes comprehensive test cases that validate:
- Algorithm consistency
- Ranking accuracy
- Score confidence
- Constraint handling

Run tests with:
```bash
python -m pytest tests/test_decision_agent.py -v -s
```

## Dependencies

- Python 3.8+
- NumPy
- SciPy
- scikit-learn

## Research Applications

This decision agent is particularly suited for:
- Research project prioritization
- Innovation assessment
- Resource allocation optimization
- Feature selection analysis

## Research Demo Tasks

Task 001: Research Text Summarization

<img width="1298" height="343" alt="image" src="https://github.com/user-attachments/assets/b156916b-d07d-49fc-a436-a789e0cd2618" />

Task 002: Research Decision Making

<img width="967" height="363" alt="image" src="https://github.com/user-attachments/assets/17047608-8d0a-40e9-8eac-9b822904678e" />

Task 003: Combined Research Analysis

<img width="968" height="524" alt="image" src="https://github.com/user-attachments/assets/9ae6ec0d-c803-4f6a-a3d0-d63a7566fe4f" />


## Author

Jyothi Sai Sravana Ravella
- LinkedIn: www.linkedin.com/in/sravanaravella
- Email: jyothisravana2005@gmail.com
