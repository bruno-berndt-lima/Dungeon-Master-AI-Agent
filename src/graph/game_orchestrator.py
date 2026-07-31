import sqlite3
from typing import Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph

from src.agents.dice_roller import DiceRollerAgent
from src.agents.dungeon_master import DungeonMaster
from src.agents.researcher import ResearcherAgent
from src.agents.supervisor import GameSupervisor
from src.graph.game_state import GameState

DEFAULT_CHECKPOINT_DB = "game_state.db"


def create_sqlite_checkpointer(db_path: str = DEFAULT_CHECKPOINT_DB) -> BaseCheckpointSaver:
    """Opens a SQLite-backed checkpointer for persistent campaign state.

    The connection deliberately outlives this call — the graph holds it for the
    life of the process. ``SqliteSaver.from_conn_string`` is a context manager
    and would close the connection on exit, which does not suit a REPL.
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn)


def create_game_graph(checkpointer: Optional[BaseCheckpointSaver] = None):
    """Creates the main game orchestration graph using agent nodes.

    There are no explicit edges. Every node returns ``Command(goto=...)`` and
    LangGraph derives the legal destinations from each ``process_task`` return
    annotation — so an annotation that disagrees with what the method actually
    returns is a bug, not a style issue.

    Passing a ``checkpointer`` makes state persist across runs; every
    ``invoke`` then needs a ``config={"configurable": {"thread_id": ...}}``.
    """
    supervisor = GameSupervisor()
    dungeon_master = DungeonMaster()
    researcher = ResearcherAgent()
    dice_roller = DiceRollerAgent()

    workflow = StateGraph(GameState)

    workflow.add_node("supervisor", supervisor.process_task)
    workflow.add_node("dungeon_master", dungeon_master.process_task)
    workflow.add_node("researcher", researcher.process_task)
    workflow.add_node("dice_roller", dice_roller.process_task)

    workflow.set_entry_point("supervisor")

    return workflow.compile(checkpointer=checkpointer)
