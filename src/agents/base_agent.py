from abc import ABC, abstractmethod
from typing import Dict, Any
from langchain_core.messages import SystemMessage
from src.utils.llm_logger import LLMLogger, LLMInteraction  
from src.graph.game_state import GameState

class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.logger = LLMLogger()
        self.initialized = False
        
    def initialize_agent(self, state: GameState):
        """Initialize agent in game state if not already present"""
        if self.agent_type not in state["game_state"]:
            # Store only essential agent info, not the entire object
            state["game_state"][self.agent_type] = {
                "type": self.agent_type,
                "initialized": True,
                "state": {}
            }
            self.initialized = True
        return state
    
    @abstractmethod
    def process_task(self, state: GameState) -> GameState:
        """Main processing method to be implemented by subclasses"""
        pass
    
    @abstractmethod
    def get_definition(self) -> str:
        """Return system prompt definition for the agent"""
        pass

    def _log_interaction(self, query: str, response: str, metadata: Dict[str, Any] = None):
        """Standard logging method"""
        interaction = LLMInteraction.create(
            agent=self.agent_type,
            query=query,
            response=response,
            metadata=metadata or {}
        )
        self.logger.log_interaction(interaction)