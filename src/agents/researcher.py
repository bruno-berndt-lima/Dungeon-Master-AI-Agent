from typing import Literal, Dict, Any
from langchain_core.messages import SystemMessage
from langgraph.graph import END
from langgraph.types import Command
from src.data.vectorstore import get_vectorstore
from src.prompts.prompts import RESEARCHER_PROMPT
from src.models.llm import create_llm
from src.agents.base_agent import BaseAgent
from src.graph.game_state import GameState

class ResearcherAgent(BaseAgent):
    """Agent that provides information about D&D rules and lore."""
    
    def __init__(self):
        super().__init__("researcher")
        self.llm = create_llm()
        self.system_prompt = RESEARCHER_PROMPT
    
    def get_definition(self) -> str:
        return "I am a researcher assistant that provides information about D&D rules, lore, monsters, spells, and game mechanics."
    
    def process_task(self, state: GameState) -> Command[Literal["supervisor"]]:
        """Retrieves and provides D&D-related information."""
        # Extract the latest message from the state
        latest_message = self._get_latest_message(state)
        
        # Prepare messages for the LLM
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": latest_message}
        ]
        
        # Get response from LLM
        try:
            response = self.llm.invoke(messages)
            
            if response is None:
                response_content = "I apologize, but I couldn't find information about that D&D topic."
            else:
                response_content = response.content if hasattr(response, "content") else str(response)
                
            # Log the interaction
            self._log_interaction(
                query=latest_message,
                response=response_content
            )
            
            # Create a new state dictionary to avoid modifying the original
            updated_state = dict(state)
            
            # Ensure there's a messages list
            if "messages" not in updated_state:
                updated_state["messages"] = []
            
            # Add the response as a properly formatted dictionary
            updated_messages = list(updated_state["messages"])
            updated_messages.append({
                "role": "assistant",
                "content": response_content,
                "name": self.agent_type
            })
            
            # Update the state with the new messages list
            updated_state["messages"] = updated_messages
            
            # Set next_agent to FINISH to terminate the conversation
            updated_state["next_agent"] = "FINISH"
            
            # Return FINISH to end the conversation after providing information
            return Command(goto="__end__", update=updated_state)
            
        except Exception as e:
            error_message = f"Error researching D&D information: {str(e)}"
            
            # Log the error
            self._log_interaction(
                query=latest_message,
                response=error_message,
                metadata={"error": str(e)}
            )
            
            # Update state with error message
            updated_state = dict(state)
            if "messages" not in updated_state:
                updated_state["messages"] = []
            
            # Add the error message as a properly formatted dictionary
            updated_messages = list(updated_state["messages"])
            updated_messages.append({
                "role": "assistant",
                "content": error_message,
                "name": self.agent_type
            })
            
            updated_state["messages"] = updated_messages
            
            # Set next_agent to FINISH even on error
            updated_state["next_agent"] = "FINISH"
            
            # Return FINISH to end the conversation
            return Command(goto="__end__", update=updated_state)
    
    def _get_latest_message(self, state: GameState) -> str:
        """Extracts the latest message content from the state."""
        if "messages" in state and state["messages"]:
            latest = state["messages"][-1]
            if isinstance(latest, dict) and "content" in latest:
                return latest["content"]
            elif hasattr(latest, "content"):
                return latest.content
            return str(latest)
        elif "current_task" in state:
            return state["current_task"]
        return ""
