"""The graph's state, and typed access to the engine models inside it.

Engine objects are stored as **plain JSON dicts**, not pydantic instances.
LangGraph's checkpointer can serialise pydantic models, but it warns that
unregistered types will be refused in a future release, and registering
every engine type is a list that would rot. Dicts survive any serialiser —
including the async saver the Discord bot needs — and the accessors below
put the types back at the boundary: `get_party(state)` returns
`Character`s, `put_party(...)` returns what to write.
"""

from typing import Annotated, Any, Dict, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from src.engine.character import Character
from src.engine.checks import PendingCheck
from src.engine.combat import Encounter


class GameState(TypedDict):
    """Two contracts worth knowing before you write a node:

    1. ``messages`` uses the ``add_messages`` reducer. A node returns **only the
       messages it produced**; LangGraph appends them to the existing history.
       Every other field replaces on write.

    2. Routing lives in ``Command(goto=...)``, not in the state. There is no
       ``next_agent`` field (PR-03), and since PR-18 no ``active_agent``
       either — ``intake`` decides where a turn goes, in code.

    ``party``, ``encounter`` and ``pending`` hold the engine's models as JSON
    dicts — see the module docstring and the accessors below.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_task: str
    tool_steps: int  # tool calls made this turn; the DM's loop is capped (PR-18)
    party: Dict[str, Dict[str, Any]]  # character name -> Character (JSON)
    encounter: Optional[Dict[str, Any]]  # Encounter (JSON) while a fight is on
    pending: Optional[Dict[str, Any]]  # PendingCheck (JSON) while a roll is awaited
    summary: str  # rolling campaign summary (PR-19)
    last_response: str


def create_default_game_state() -> GameState:
    """Creates a default game state with initial values."""
    return GameState(
        messages=[],
        current_task="",
        tool_steps=0,
        party={},
        encounter=None,
        pending=None,
        summary="",
        last_response="",
    )


# --- typed access to the engine models -------------------------------------------

def get_party(state: GameState) -> Dict[str, Character]:
    return {name: Character.model_validate(data) for name, data in (state.get("party") or {}).items()}


def put_party(party: Dict[str, Character]) -> Dict[str, Dict[str, Any]]:
    return {name: character.model_dump(mode="json") for name, character in party.items()}


def get_encounter(state: GameState) -> Optional[Encounter]:
    data = state.get("encounter")
    return Encounter.model_validate(data) if data else None


def put_encounter(encounter: Optional[Encounter]) -> Optional[Dict[str, Any]]:
    return encounter.model_dump(mode="json") if encounter else None


def get_pending(state: GameState) -> Optional[PendingCheck]:
    data = state.get("pending")
    return PendingCheck.model_validate(data) if data else None


def put_pending(pending: Optional[PendingCheck]) -> Optional[Dict[str, Any]]:
    return pending.model_dump(mode="json") if pending else None
