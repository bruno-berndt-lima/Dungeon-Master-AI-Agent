from typing import Annotated, Any, Dict, List, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from src.actors.npc import NPC
from src.actors.player import Player


class GameState(TypedDict):
    """Represents the current state of the game.

    Two contracts worth knowing before you write a node:

    1. ``messages`` uses the ``add_messages`` reducer. A node returns **only the
       messages it produced**; LangGraph appends them to the existing history.
       Returning the whole history duplicates it — the reducer appends, it does
       not replace. Every other field replaces on write.

    2. Routing lives in ``Command(goto=...)``, not in the state. There is no
       ``next_agent`` field: it duplicated the real routing channel, the two
       could disagree, and ``main.py`` treated ``next_agent == "FINISH"`` as a
       signal to exit the REPL — so the session ended after every answer the
       researcher gave.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_task: str
    active_agent: str
    game_state: Dict[str, Any]
    players: Dict[str, Player]
    npcs: Dict[str, NPC]
    current_speaker: str
    turn_order: List[str]
    last_response: str
    requires_player_input: bool


def create_default_game_state() -> GameState:
    """Creates a default game state with initial values.

    ``game_state`` is a dict and must stay one — ``BaseAgent.initialize_agent``
    does a key lookup against it, which silently degrades to a substring check
    if it is ever replaced with a string.
    """
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
    )
