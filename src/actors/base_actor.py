from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List
from langchain_core.messages import BaseMessage

@dataclass
class Actor(ABC):
    """Base class for all actors in the game (Players, NPCs, etc.)"""
    id: str
    name: str
    type: str  # "player", "npc", "dm"
    description: Optional[str] = None
    
    @abstractmethod
    def can_act(self) -> bool:
        """Determines if the actor can take actions"""
        pass
    
    @abstractmethod
    def process_message(self, message: BaseMessage) -> Optional[BaseMessage]:
        """Process incoming messages"""
        pass 