from .base_actor import Actor
from langchain_core.messages import BaseMessage
from typing import Optional

class Player(Actor):
    """Represents a human player in the game"""
    def __init__(self, id: str, name: str):
        super().__init__(
            id=id,
            name=name,
            type="player",
            description=f"Human player: {name}"
        )
        
    def can_act(self) -> bool:
        """Players can always act (they're human)"""
        return True
        
    def process_message(self, message: BaseMessage) -> Optional[BaseMessage]:
        """Players process messages through the UI"""
        # This would be handled by your UI layer
        return None 