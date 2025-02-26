from .base_actor import Actor
from dataclasses import dataclass
from langchain_core.messages import BaseMessage
from typing import Dict, Optional

@dataclass
class NPCStats:
    """Basic NPC statistics"""
    health: int
    strength: int
    dexterity: int
    # Add other D&D stats as needed

class NPC(Actor):
    """Represents an AI-controlled NPC"""
    def __init__(self, id: str, name: str, stats: NPCStats, personality: str):
        super().__init__(
            id=id,
            name=name,
            type="npc",
            description=f"NPC: {name} - {personality}"
        )
        self.stats = stats
        self.personality = personality
        self.is_active = True
        self.state: Dict = {}  # Store NPC state (location, inventory, etc.)
        
    def can_act(self) -> bool:
        """NPCs can act if they're active and alive"""
        return self.is_active and self.stats.health > 0
        
    def process_message(self, message: BaseMessage) -> Optional[BaseMessage]:
        """Process messages using NPC's AI personality"""
        # TODO: Implement NPC AI response generation
        return None 