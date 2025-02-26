from typing import TypedDict, Literal
from langgraph.graph import MessagesState, END
from langgraph.types import Command
from src.prompts.prompts import SUPERVISOR_PROMPT
from src.models.llm import create_llm
from src.agents.base_agent import BaseAgent
from src.graph.game_state import GameState

AGENT_TYPES = ["dungeon_master", "researcher", "dice_roller"]

ROUTING_OPTIONS = AGENT_TYPES + ["FINISH"]

class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal[*ROUTING_OPTIONS]

class State(MessagesState):
    next: str

class GameSupervisor(BaseAgent):
    """Supervisor class that manages routing between game agents."""
    
    def __init__(self):
        super().__init__("supervisor")
        self.llm = create_llm()
        self.system_prompt = SUPERVISOR_PROMPT

    def get_definition(self) -> str:
        return self.system_prompt

    def process_task(self, state: GameState) -> Command[Literal[*AGENT_TYPES, "__end__"]]:
        # If there's an explicit FINISH signal, end the process
        if state.get("next") == "FINISH" or state.get("next_agent") == "FINISH":
            return Command(goto=END, update=state)
        
        # Extract the current task
        current_task = state.get("current_task", "")
        
        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        
        # Add the latest message from the user
        if "messages" in state and state["messages"]:
            for message in state["messages"]:
                if isinstance(message, dict):
                    messages.append(message)
                else:
                    content = message.content if hasattr(message, "content") else str(message)
                    messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": current_task})
        
        try:
            # Send to LLM for routing decision
            response = self.llm.invoke(messages)
            
            # Parse the response
            if response is None:
                goto = "researcher"  # Default if no response
            elif hasattr(response, "content"):
                response_text = response.content.strip().lower()
                
                # Parse the response to extract the agent name
                if "dice_roller" in response_text:
                    goto = "dice_roller"
                elif "dungeon_master" in response_text:
                    goto = "dungeon_master"
                elif "researcher" in response_text:
                    goto = "researcher"
                elif "finish" in response_text:
                    goto = "FINISH"
                else:
                    goto = "researcher"  # Default if no match
            else:
                goto = response["next"] if isinstance(response, dict) and "next" in response else "researcher"
                
        except Exception as e:
            # If there's any error in getting or processing the LLM response,
            # default to researcher
            goto = "researcher"
            self._log_interaction(
                query=str(messages),
                response=f"Error: {str(e)}. Defaulting to researcher",
                metadata={"error": str(e), "next_agent": goto}
            )
        
        # Convert FINISH to END for langgraph
        if goto == "FINISH":
            goto = END
        
        # Update state with next decision
        updated_state = dict(state)
        updated_state["next_agent"] = goto
        
        # Log the interaction
        self._log_interaction(
            query=str(messages),
            response=str(goto),
            metadata={"next_agent": goto}
        )
        
        return Command(goto=goto, update=updated_state)
    
        

