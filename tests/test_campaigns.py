"""Campaign resume (PR-13): naming, listing, and picking up a thread.

A stand-in node plays the DM, so nothing here needs a model — but the
checkpointer and the graph are real.
"""

import sqlite3

import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("langgraph", reason="full dependency stack not installed")

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402
from langgraph.graph import END, StateGraph  # noqa: E402

from main import build_parser, format_campaigns  # noqa: E402
from src.graph.campaigns import (  # noqa: E402
    THREAD_ID_LENGTH,
    is_new_campaign,
    list_campaigns,
    new_thread_id,
    recap,
    seed_turn,
    summarize,
)
from src.graph.game_state import GameState, create_default_game_state  # noqa: E402


def _dm_stub(state):
    """Narrates a fixed line and moves the party somewhere."""
    return {
        "messages": [AIMessage(content="The door creaks open.", name="dungeon_master")],
        "game_state": {"location": "the crypt"},
    }


@pytest.fixture
def graph_and_saver(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "cp.db"), check_same_thread=False)
    saver = SqliteSaver(conn)
    workflow = StateGraph(GameState)
    workflow.add_node("dungeon_master", _dm_stub)
    workflow.set_entry_point("dungeon_master")
    workflow.add_edge("dungeon_master", END)
    graph = workflow.compile(checkpointer=saver)
    try:
        yield graph, saver
    finally:
        conn.close()


def _play(graph, thread_id, *inputs):
    config = {"configurable": {"thread_id": thread_id}}
    for text in inputs:
        values = dict(graph.get_state(config).values or {})
        turn = seed_turn({"messages": [HumanMessage(content=text)], "current_task": text}, values)
        graph.invoke(turn, config=config)
    return config


# --- naming -----------------------------------------------------------------

def test_new_thread_ids_are_short_and_distinct():
    ids = {new_thread_id() for _ in range(50)}
    assert len(ids) == 50
    assert all(len(i) == THREAD_ID_LENGTH and i.isalnum() for i in ids)


# --- new vs resumed ---------------------------------------------------------

def test_an_unseen_thread_is_new(graph_and_saver):
    graph, _ = graph_and_saver
    values = graph.get_state({"configurable": {"thread_id": "nope"}}).values
    assert is_new_campaign(values)


def test_a_played_thread_is_not_new(graph_and_saver):
    graph, _ = graph_and_saver
    config = _play(graph, "c1", "I open the door")
    assert not is_new_campaign(graph.get_state(config).values)


def test_the_first_turn_of_a_new_thread_carries_the_default_state():
    turn = seed_turn({"messages": [HumanMessage(content="hi")], "current_task": "hi"}, {})
    defaults = create_default_game_state()
    assert set(defaults) <= set(turn)
    assert turn["game_state"] == {}
    assert turn["current_task"] == "hi"


def test_a_resumed_turn_carries_only_the_message():
    """Merging the defaults back in would replace the stored game_state with {}."""
    existing = {"messages": [HumanMessage(content="earlier")], "game_state": {"location": "the crypt"}}
    turn = seed_turn({"messages": [HumanMessage(content="hi")], "current_task": "hi"}, existing)
    assert set(turn) == {"messages", "current_task"}


def test_resuming_keeps_the_world_state(graph_and_saver):
    """The bug this PR fixes: main.py used to re-seed on its first turn."""
    graph, _ = graph_and_saver
    config = _play(graph, "c1", "I open the door")
    assert graph.get_state(config).values["game_state"]["location"] == "the crypt"

    # A new process, same thread: the resumed turn must not carry defaults.
    values = dict(graph.get_state(config).values)
    turn = seed_turn({"messages": [HumanMessage(content="I look around")], "current_task": "..."}, values)
    assert "game_state" not in turn

    graph.invoke(turn, config=config)
    after = graph.get_state(config).values
    assert after["game_state"]["location"] == "the crypt"
    assert [m.content for m in after["messages"] if isinstance(m, HumanMessage)] == [
        "I open the door", "I look around",
    ]


# --- listing ----------------------------------------------------------------

def test_listing_shows_every_thread_with_its_turns(graph_and_saver):
    graph, saver = graph_and_saver
    _play(graph, "alpha", "one", "two", "three")
    _play(graph, "beta", "solo")

    campaigns = {c.thread_id: c for c in list_campaigns(saver)}
    assert set(campaigns) == {"alpha", "beta"}
    assert campaigns["alpha"].turns == 3
    assert campaigns["beta"].turns == 1
    assert campaigns["alpha"].location == "the crypt"
    assert campaigns["alpha"].last_prompt == "three"


def test_listing_is_newest_first(graph_and_saver):
    graph, saver = graph_and_saver
    _play(graph, "older", "x")
    _play(graph, "newer", "y")
    assert [c.thread_id for c in list_campaigns(saver)] == ["newer", "older"]

    _play(graph, "older", "z")  # activity moves it to the top
    assert [c.thread_id for c in list_campaigns(saver)] == ["older", "newer"]


def test_listing_an_empty_database_is_empty(graph_and_saver):
    _, saver = graph_and_saver
    assert list_campaigns(saver) == []


def test_summary_tolerates_a_bare_state():
    summary = summarize("t", {})
    assert (summary.turns, summary.location, summary.last_prompt) == (0, "", "")


# --- recap ------------------------------------------------------------------

def test_recap_puts_the_player_back_in_the_scene(graph_and_saver):
    graph, _ = graph_and_saver
    config = _play(graph, "c1", "I open the door", "I step in")
    text = recap(graph.get_state(config).values)
    assert "2 turns" in text
    assert "the crypt" in text
    assert "The door creaks open." in text


def test_recap_is_empty_for_a_new_campaign():
    assert recap({}) == ""


def test_recap_ignores_other_agents_output():
    values = {
        "messages": [
            HumanMessage(content="roll"),
            AIMessage(content="🎲 12", name="dice_roller"),
        ],
        "game_state": {},
    }
    assert "🎲" not in recap(values)


# --- the CLI surface --------------------------------------------------------

def test_parser_defaults_to_a_new_campaign():
    args = build_parser().parse_args([])
    assert args.thread is None and args.list is False


def test_parser_accepts_thread_and_list():
    assert build_parser().parse_args(["--thread", "tuesday"]).thread == "tuesday"
    assert build_parser().parse_args(["--list"]).list is True


def test_format_campaigns_names_each_thread(graph_and_saver):
    graph, saver = graph_and_saver
    _play(graph, "alpha", "one")
    text = format_campaigns(list_campaigns(saver))
    assert "alpha" in text and "the crypt" in text and "--thread" in text


def test_format_campaigns_when_there_are_none():
    assert "No campaigns yet" in format_campaigns([])
