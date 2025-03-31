from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import MessagesState, END
from langgraph.types import Command
from src.agents.base_agent import BaseAgent
from src.models.llm import create_llm
from src.graph.game_state import GameState
from src.prompts.prompts import DUNGEON_MASTER_PROMPT
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

class DungeonMaster(BaseAgent):
    """Dungeon Master class that manages game interactions."""
    
    def __init__(self):
        super().__init__("dungeon_master")
        self.llm = create_llm()
        self.system_prompt = DUNGEON_MASTER_PROMPT

    def get_definition(self) -> str:
        return self.system_prompt

    def process_task(self, state: GameState) -> Command[Literal["supervisor"]]:
        """Processes a game-related task."""
        pass
