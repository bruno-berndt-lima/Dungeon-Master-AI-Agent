from src.agents.researcher import ResearcherAgent
from src.graph.game_state import GameState

def test_researcher():
    # Create an instance of the ResearcherAgent
    researcher = ResearcherAgent()
    
    # Test cases - different research questions about D&D
    test_cases = [
        "What are the rules for making a Wisdom saving throw?",
        "Tell me about the Fireball spell",
        "How does sneak attack work for rogues?",
        "What are the standard rules for long rest?",
        "What is the AC of a Basilisk?"        
    ]
    
    # Process each test case
    for test_input in test_cases:
        print("\n" + "="*80)
        print(f"Query: \"{test_input}\"")
        
        # Create a simple mock state with just the test input
        mock_state = GameState(
            messages=[{"role": "user", "content": test_input}],
            current_task=test_input
        )
        
        # Process the state with the researcher
        try:
            result = researcher.process_task(mock_state)
            
            # Print just the answer
            if hasattr(result, "update") and "messages" in result.update:
                messages = result.update.get("messages", [])
                if messages and isinstance(messages[-1], dict):
                    print("\nAnswer:")
                    print(messages[-1].get("content", "No answer found"))
                else:
                    print("\nNo readable answer found in response")
            else:
                print("\nUnexpected response format")
        
        except Exception as e:
            print(f"\nError: {str(e)}")
            
        print("="*80)

if __name__ == "__main__":
    test_researcher() 