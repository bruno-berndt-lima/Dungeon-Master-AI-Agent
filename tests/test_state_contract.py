"""Tests for the GameState contract introduced in PR-03.

These exercise the reducer and the checkpointer directly, with a stand-in node
instead of a real agent — so they need no model daemon, but they do need the
dependency stack and Python 3.11+.
"""

import sqlite3
import sys

import pytest

pytestmark = pytest.mark.integration

if sys.version_info < (3, 11):
    pytest.skip("src/agents/supervisor.py needs PEP 646 syntax", allow_module_level=True)

pytest.importorskip("langgraph", reason="full dependency stack not installed")

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402
from langgraph.graph import END, StateGraph  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402

from src.graph.game_state import GameState, create_default_game_state  # noqa: E402


def _one_node_graph(node, checkpointer=None):
    workflow = StateGraph(GameState)
    workflow.add_node("echo", node)
    workflow.set_entry_point("echo")
    workflow.add_edge("echo", END)
    return workflow.compile(checkpointer=checkpointer)


# --------------------------------------------------------------------------- #
# The reducer
# --------------------------------------------------------------------------- #

def test_returning_a_delta_appends_exactly_one_message():
    graph = _one_node_graph(
        lambda state: {"messages": [AIMessage(content="ack", name="echo")]}
    )
    result = graph.invoke({"messages": [HumanMessage(content="hello")]})
    assert [m.content for m in result["messages"]] == ["hello", "ack"]


def test_returning_the_whole_history_is_deduped_by_id():
    """add_messages merges on message id, not position.

    Messages read back out of state already carry ids, so re-returning them is
    safe — this is why the pre-PR-03 whole-list pattern did not corrupt history.
    Returning deltas is still preferred (smaller payloads, idiomatic), but it is
    a style choice here, not a correctness fix.
    """
    def whole_list_node(state):
        history = list(state["messages"])
        history.append(AIMessage(content="ack", name="echo"))
        return {"messages": history}

    result = _one_node_graph(whole_list_node).invoke(
        {"messages": [HumanMessage(content="hello")]}
    )
    assert [m.content for m in result["messages"]] == ["hello", "ack"]


def test_rebuilding_messages_without_ids_duplicates_history():
    """The real hazard: constructing fresh message objects drops their ids, so
    the reducer cannot match them against what is already in state and appends
    them a second time."""
    def rebuilding_node(state):
        history = [HumanMessage(content=m.content) for m in state["messages"]]
        history.append(AIMessage(content="ack", name="echo"))
        return {"messages": history}

    result = _one_node_graph(rebuilding_node).invoke(
        {"messages": [HumanMessage(content="hello")]}
    )
    assert [m.content for m in result["messages"]] == ["hello", "hello", "ack"]


def test_add_messages_coerces_plain_dicts_to_message_objects():
    """Why BaseAgent._get_latest_message no longer needs a dict branch."""
    merged = add_messages([], [{"role": "user", "content": "roll a d20"}])
    assert isinstance(merged[0], HumanMessage)


# --------------------------------------------------------------------------- #
# The checkpointer
# --------------------------------------------------------------------------- #

def test_checkpointer_persists_history_across_invocations(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "cp.db"), check_same_thread=False)
    try:
        graph = _one_node_graph(
            lambda state: {"messages": [AIMessage(content="ack", name="echo")]},
            checkpointer=SqliteSaver(conn),
        )
        config = {"configurable": {"thread_id": "campaign-1"}}

        first = graph.invoke({"messages": [HumanMessage(content="turn one")]}, config=config)
        assert len(first["messages"]) == 2

        # Only the new message is passed; the rest is restored from the checkpoint.
        second = graph.invoke({"messages": [HumanMessage(content="turn two")]}, config=config)
        assert [m.content for m in second["messages"]] == [
            "turn one", "ack", "turn two", "ack",
        ]
    finally:
        conn.close()


def test_threads_are_isolated(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "cp.db"), check_same_thread=False)
    try:
        graph = _one_node_graph(
            lambda state: {"messages": [AIMessage(content="ack", name="echo")]},
            checkpointer=SqliteSaver(conn),
        )
        graph.invoke(
            {"messages": [HumanMessage(content="campaign A")]},
            config={"configurable": {"thread_id": "a"}},
        )
        other = graph.invoke(
            {"messages": [HumanMessage(content="campaign B")]},
            config={"configurable": {"thread_id": "b"}},
        )
        assert [m.content for m in other["messages"]] == ["campaign B", "ack"]
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# game_state stays a dict
# --------------------------------------------------------------------------- #

def test_game_state_survives_a_turn_as_a_dict():
    """KNOWN_ISSUES #4: main.py used to overwrite this with the string
    "initialized", turning BaseAgent.initialize_agent's key lookup into a
    substring check."""
    graph = _one_node_graph(lambda state: {"game_state": {**state["game_state"], "seen": True}})
    result = graph.invoke(create_default_game_state())
    assert isinstance(result["game_state"], dict)
    assert result["game_state"]["seen"] is True
