from typing import TypedDict, Sequence, Dict, List
from langchain_core.messages import BaseMessage
from src.actors.player import Player
from src.actors.npc import NPC

class GameState(TypedDict):
    """Represents the current state of the game."""
    messages: Sequence[BaseMessage]
    current_task: str
    active_agent: str
    game_state: dict
    players: Dict[str, Player]
    npcs: Dict[str, NPC]
    current_speaker: str
    turn_order: List[str]
    last_response: str
    requires_player_input: bool
    next_agent: str

def create_default_game_state() -> GameState:
    """Creates a default game state with initial values."""
    return GameState(
        messages=[],
        current_task="",
        active_agent="supervisor",  
        game_state={},
        players={},
        npcs={},
        current_speaker="",
        turn_order=[],
        last_response="",
        requires_player_input=False,
        next_agent="supervisor"
    )
