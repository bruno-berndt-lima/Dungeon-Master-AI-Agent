"""The graph. Five nodes, no LLM router, no edges but `Command`s.

    intake ──▶ dungeon_master ⇄ tools ──▶ memory ──▶ END
       ├─────▶ researcher ─────────────────────────▶ END        (/rules)
       └─────▶ END                                              (/roll, /join, dice, help)

`intake` decides in code where a turn goes. The Dungeon Master loops with
`tools` at most `MAX_TOOL_STEPS` times and always ends with narration, after
which `memory` folds old messages into the campaign journal when needed.
"""

import sqlite3
from typing import Any, Literal, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
from langgraph.types import Command

from src.agents.dungeon_master import DungeonMaster
from src.agents.memory import Memory
from src.agents.researcher import ResearcherAgent
from src.graph.game_state import GameState
from src.graph.intake import intake
from src.utils.dice import RandomSource

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


def create_game_graph(
    checkpointer: Optional[BaseCheckpointSaver] = None,
    dm_llm: Any = None,
    researcher: Optional[ResearcherAgent] = None,
    rng: Optional[RandomSource] = None,
    memory_llm: Any = None,
):
    """Build and compile the graph.

    ``dm_llm``, ``memory_llm`` and ``researcher`` exist so tests can drive the
    whole graph with stubs and no daemon; ``rng`` makes every roll in a
    session replay. Passing a ``checkpointer`` makes state persist across
    runs; every ``invoke`` then needs ``config={"configurable": {"thread_id": ...}}``.
    """
    dungeon_master = DungeonMaster(llm=dm_llm, rng=rng)
    memory = Memory(llm=memory_llm)
    researcher = researcher or ResearcherAgent()

    def intake_node(state: GameState) -> Command[Literal["dungeon_master", "researcher", "__end__"]]:
        return intake(state, rng=rng)

    workflow = StateGraph(GameState)
    workflow.add_node("intake", intake_node)
    workflow.add_node("dungeon_master", dungeon_master.process_task)
    workflow.add_node("tools", dungeon_master.run_tools)
    workflow.add_node("memory", memory.process_task)
    workflow.add_node("researcher", researcher.process_task)
    workflow.set_entry_point("intake")

    return workflow.compile(checkpointer=checkpointer)
