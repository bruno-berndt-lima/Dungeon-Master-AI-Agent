# minimal_main.py

from src.graph.game_orchestrator import create_game_graph
from src.graph.game_state import create_default_game_state
import sys
import traceback

def main():
    # Initialize the game orchestrator
    try:
        game_graph = create_game_graph()
        print("✅ Game graph created successfully.")
    except Exception as e:
        print(f"❌ Failed to create game graph: {e}")
        sys.exit(1)
    
    print("🎲 Initializing D&D adventure...\n")
    print("🛡️ D&D Assistant ready! Type 'quit' or 'exit' to end the session.\n")
    
    # Create default game state that's compatible with the graph
    try:
        state = create_default_game_state()
        print("✅ Default game state initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to create default game state: {e}")
        sys.exit(1)
    
    # Add custom state variables
    state.update({
        "current_task": "",
        "messages": [],
        "active_agent": "supervisor", 
        "game_state": "initialized",
        "players": {},
        "npcs": {},
        "turn_order": [],
        "next_agent": "supervisor"
    })
    
    # Debug: Print the initial state
    print("\nInitial GameState:")
    for key, value in state.items():
        print(f"{key}: {value}")
    
    # Main interaction loop
    while True:
        try:
            # Get user input
            user_input = input("🎲 Ask a D&D question: ")
            
            # Check for exit command
            if user_input.lower() in ["quit", "exit"]:
                print("Ending D&D session. Farewell, adventurer!")
                break
            
            # Add user message to the state
            state["messages"].append({
                "role": "user",
                "content": user_input
            })
            
            # Set the current task to user input
            state["current_task"] = user_input
            
            # Debug: Print state before invoking the graph
            print("\nState before invoking the graph:")
            for key, value in state.items():
                print(f"{key}: {value}")
            
            # Process the task through the graph
            state = game_graph.invoke(state)
            
            # Reset current task after processing
            state["current_task"] = ""
            
            # Check if the session should terminate
            if state.get("next_agent") == "FINISH":
                print("🔚 D&D session has been terminated by the assistant.")
                break
            
        except KeyError as ke:
            print(f"\n❌ KeyError: Missing key {ke}")
            traceback.print_exc()
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            traceback.print_exc()

        # Print GameState variables safely
        print("\n🔄 Current GameState variables:")
        for key, value in state.items():
            print(f" - {key}: {value}")
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()