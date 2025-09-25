import unittest
from yara.agents.decision_agent import DecisionAgent

class TestDecisionAgent(unittest.TestCase):
    def setUp(self):
        self.agent = DecisionAgent()
        
        # Simple test data
        self.test_task = {
            "type": "decision",
            "data": [
                {
                    "name": "Option A",
                    "novelty": 0.8,  # High novelty
                    "research_impact": 85,  # High impact
                    "implementation_complexity": "medium",
                    "features": ["core", "innovative"],
                    "price": 100
                },
                {
                    "name": "Option B",
                    "novelty": 0.4,  # Medium novelty
                    "research_impact": 60,  # Medium impact
                    "implementation_complexity": "low",
                    "features": ["core"],
                    "price": 50
                },
                {
                    "name": "Option C",
                    "novelty": 0.2,  # Low novelty
                    "research_impact": 30,  # Low impact
                    "implementation_complexity": "high",
                    "features": ["experimental"],
                    "price": 200
                }
            ],
            "user_preferences": {
                "implementation_complexity": "medium",  # Prefer medium complexity
                "required_features": ["core"],  # Must have core feature
                "max_price": 150  # Budget constraint
            },
            "criteria": {
                "novelty": 0.4,  # 40% weight on novelty
                "research_impact": 0.6  # 60% weight on research impact
            }
        }

    def test_decision_agent(self):
        print("\n" + "="*80)
        print("YARA Decision Agent Evaluation Results")
        print("="*80)

        print("\n1. Experimental Setup")
        print("-"*50)
        print("\n1.1 Available Options Analysis:")
        for opt in self.test_task["data"]:
            print(f"\nOption: {opt['name']}")
            print(f"├── Innovation Metrics:")
            print(f"│   ├── Novelty Score: {opt['novelty']:.2f}")
            print(f"│   └── Research Impact: {opt['research_impact']}/100")
            print(f"├── Implementation Characteristics:")
            print(f"│   ├── Complexity Level: {opt['implementation_complexity']}")
            print(f"│   └── Feature Set: {', '.join(opt['features'])}")
            print(f"└── Resource Requirements:")
            print(f"    └── Cost: ${opt['price']}")

        print("\n1.2 Decision Constraints:")
        prefs = self.test_task["user_preferences"]
        print("\nUser-Defined Parameters:")
        print(f"├── Complexity Preference: {prefs['implementation_complexity']}")
        print(f"├── Essential Features: {', '.join(prefs['required_features'])}")
        print(f"└── Budget Constraint: ${prefs['max_price']}")

        print("\n1.3 Evaluation Criteria Weights:")
        print(f"├── Novelty Weight: {self.test_task['criteria']['novelty']*100}%")
        print(f"└── Research Impact Weight: {self.test_task['criteria']['research_impact']*100}%")

        print("\n2. Decision Process Execution")
        print("-"*50)
        print("\nInitiating decision analysis algorithm...")
        result = self.agent.execute(self.test_task)
        
        print("\n3. Analysis Results")
        print("-"*50)
        
        # Verify and analyze results
        self.assertEqual(result["type"], "recommendation")
        self.assertTrue("recommendations" in result)
        self.assertTrue(len(result["recommendations"]) > 0)
        
        print("\n3.1 Ranked Recommendations:")
        for i, rec in enumerate(result["recommendations"], 1):
            score_percentage = rec["score"] * 100
            print(f"\nRank {i}: {rec['option']['name']}")
            print(f"├── Evaluation Score: {score_percentage:.1f}%")
            print("├── Decision Factors:")
            reasons = rec["reasoning"].split(" | ")
            for reason in reasons:
                print(f"│   └── {reason}")
            
            # Detailed metrics
            opt = rec["option"]
            print("└── Metric Analysis:")
            print(f"    ├── Innovation Index: {opt['novelty']*100:.1f}%")
            print(f"    ├── Research Impact Factor: {opt['research_impact']}/100")
            print(f"    ├── Implementation Level: {opt['implementation_complexity']}")
            print(f"    └── Resource Efficiency: ${opt['price']}")

        print("\n4. Validation")
        print("-"*50)
        
        # Validate top recommendation
        first_rec = result["recommendations"][0]
        self.assertEqual(first_rec["option"]["name"], "Option A", 
                        "Highest novelty and impact option should be ranked first")
        self.assertTrue(first_rec["score"] > 0.7, 
                        "Top recommendation should have a strong confidence score")
        
        print("\nValidation Results:")
        print("├── Algorithm Consistency: ✓ Passed")
        print("├── Ranking Accuracy: ✓ Verified")
        print("└── Score Confidence: ✓ Confirmed")

        print("\n5. Conclusion")
        print("-"*50)
        top_rec = result["recommendations"][0]
        print(f"\nOptimal Selection: {top_rec['option']['name']}")
        print(f"Confidence Level: {top_rec['score']*100:.1f}%")
        print("\nKey Decision Factors:")
        for reason in top_rec["reasoning"].split(" | "):
            print(f"└── {reason}")

if __name__ == '__main__':
    unittest.main()
