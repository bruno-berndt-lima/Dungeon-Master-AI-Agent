from src.agents.dice_roller import DiceRollerAgent
from src.graph.game_state import GameState
import json

def test_dice_roller():
    # Create an instance of the DiceRollerAgent
    dice_agent = DiceRollerAgent()
    
    # Test cases - different dice rolling requests
    test_cases = [
        "Roll a d20 for perception",
        "I need to roll 2d6 plus 3 for damage",
        "Roll with advantage on my attack",
        "Roll 2d20 take the lowest for stealth",
        "Roll 2d8 + 1d6 for Eldritch Blast",
        "Roll a d100 for random treasure",
        "Roll 3d4 for Magic Missile damage",
        "Roll d20+5 for Athletics check",
        "Roll 4d6 + 2 for damage"
    ]
    
    # Process each test case
    for test_input in test_cases:
        print("\n" + "="*50)
        print(f"Test Input: \"{test_input}\"")
        
        # Create a simple mock state with just the test input
        mock_state = GameState(
            messages=[{"role": "user", "content": test_input}],
            current_task=test_input
        )
        
        # Process the state with the dice roller
        result = dice_agent.process_task(mock_state)
        
        # Print the result
        print("\nDice Roller Result:")
        if isinstance(result, dict) and "messages" in result:
            print(result["messages"][-1]["content"])
        elif hasattr(result, "update") and "messages" in result.update:
            messages = result.update.get("messages", [])
            if messages and isinstance(messages[-1], dict):
                print(messages[-1].get("content", "No content found"))
            else:
                print("No readable message found in result")
        else:
            print("Result format not as expected:", result)
        
        # Print the parsed request if available
        print("\nParsed Request:")
        # We don't have access to the log, so just print what we know
        print(f"Direction: {getattr(result, 'goto', 'unknown')}")
        print("Command result successfully returned")
            
        print("="*50)

if __name__ == "__main__":
    test_dice_roller()
