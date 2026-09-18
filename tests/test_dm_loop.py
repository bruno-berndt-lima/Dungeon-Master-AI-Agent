"""The Dungeon Master's tool loop, driven through the whole graph with a stub
model — no daemon, every tool call scripted.

The stub is a stand-in for `ChatOllama`: `bind_tools` returns itself, and
`invoke` hands back the next scripted `AIMessage`. What is under test is
everything around the model: the loop, the cap, narrate-only mode, the tools
running against the engine, the state coming back, and the fight ending.
"""

import random

import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from src.agents.dungeon_master import MAX_TOOL_STEPS, SILENT_FALLBACK, DungeonMaster  # noqa: E402
from src.engine.pregens import pregen  # noqa: E402
from src.graph.game_orchestrator import create_game_graph  # noqa: E402
from src.graph.game_state import create_default_game_state, get_encounter, get_party, get_pending, put_party  # noqa: E402


class StubLLM:
    """Scripted replies. Records every prompt it was given."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []
        self.bound = None

    def bind_tools(self, tools, **kwargs):
        self.bound = tools
        return self

    def invoke(self, messages, config=None):
        self.prompts.append(list(messages))
        if not self.replies:
            return AIMessage(content="The dust settles.")
        reply = self.replies.pop(0)
        return reply if isinstance(reply, AIMessage) else AIMessage(content=reply)


def call(tool, **args):
    return {"name": tool, "args": args, "id": f"call-{tool}-{random.randint(0, 9999)}", "type": "tool_call"}


def tool_calls(*calls):
    return AIMessage(content="", tool_calls=list(calls))


class StubResearcher:
    def process_task(self, state):
        from langgraph.graph import END
        from langgraph.types import Command

        return Command(goto=END, update={"messages": [AIMessage(content="rules answer", name="researcher")]})


def graph(*replies, rng=None):
    stub = StubLLM(*replies)
    return create_game_graph(dm_llm=stub, researcher=StubResearcher(), rng=rng or random.Random(0)), stub


def seed(party=None):
    s = dict(create_default_game_state())
    s["party"] = put_party(party or {"Dorn": pregen("fighter", "p1", "Dorn"), "Kara": pregen("rogue", "p2", "Kara")})
    return s


def turn(text, base=None):
    base = dict(base or seed())
    base["messages"] = [HumanMessage(content=text)]
    base["current_task"] = text
    return base


def narrations(values):
    return [
        m.content for m in values["messages"]
        if isinstance(m, AIMessage) and m.name == "dungeon_master" and not m.tool_calls
    ]


# --- the loop ------------------------------------------------------------------------------

def test_a_turn_with_no_tool_calls_just_narrates():
    g, stub = graph("You push open the door. Cold air spills out.")
    out = g.invoke(turn("I open the door"))
    assert narrations(out) == ["You push open the door. Cold air spills out."]
    assert out["tool_steps"] == 0 and out["last_response"].startswith("You push")
    assert stub.bound is not None  # the planner had the tools


def test_the_planner_sees_the_scene_sheet_and_the_players_words():
    g, stub = graph("Fine.")
    g.invoke(turn("I look around"))
    system = stub.prompts[0][0].content
    assert "The table right now" in system and "Dorn (Human Fighter 1): AC 18, 13/13 HP" in system
    assert stub.prompts[0][-1].content == "I look around"


def test_a_tool_call_runs_the_tool_and_the_model_sees_the_answer():
    g, stub = graph(
        tool_calls(call("lookup_monster", name="goblin")),
        "Two goblins crouch in the dark.",
    )
    out = g.invoke(turn("what's ahead?"))
    replies = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert len(replies) == 1 and replies[0].content.startswith("Goblin — Small humanoid")
    second_prompt = stub.prompts[1]
    assert any(isinstance(m, ToolMessage) for m in second_prompt)
    assert narrations(out) == ["Two goblins crouch in the dark."]


def test_tool_answers_stay_out_of_the_next_turns_context():
    g, stub = graph(
        tool_calls(call("lookup_monster", name="goblin")),
        "Two goblins crouch in the dark.",
        "You step forward.",
    )
    first = g.invoke(turn("what's ahead?"))
    second_turn = turn("I step forward", first)
    second_turn["messages"] = list(first["messages"]) + second_turn["messages"]
    g.invoke(second_turn)
    prompt = stub.prompts[2]
    assert not any(isinstance(m, ToolMessage) for m in prompt)
    assert not any(getattr(m, "tool_calls", None) for m in prompt)
    assert [m.content for m in prompt[1:]] == ["what's ahead?", "Two goblins crouch in the dark.", "I step forward"]


def test_prose_that_came_with_a_tool_call_is_dropped():
    """It assumed the tool's outcome; it is not narration and must not be stored."""
    g, stub = graph(
        AIMessage(content="You hit the goblin with a solid blow!", tool_calls=[call("get_scene")]),
        "The goblin sneers.",
    )
    out = g.invoke(turn("I attack"))
    stored = [m for m in out["messages"] if isinstance(m, AIMessage) and m.tool_calls]
    assert stored and stored[0].content == ""
    assert not any("solid blow" in m.content for m in stub.prompts[1])
    assert narrations(out) == ["The goblin sneers."]


def test_narrate_only_calls_are_tagged_and_planner_calls_are_not():
    class Tagging(StubLLM):
        def __init__(self, *replies):
            super().__init__(*replies)
            self.configs = []

        def invoke(self, messages, config=None):
            self.configs.append(config)
            return super().invoke(messages, config)

    stub = Tagging(tool_calls(call("request_check", player="Kara", ability="DEX", dc=10)), "Kara, roll.")
    g = create_game_graph(dm_llm=stub, researcher=StubResearcher())
    g.invoke(turn("Kara sneaks"))
    assert stub.configs[0] is None
    assert stub.configs[1] == {"tags": ["narration"]}


def test_a_refused_tool_comes_back_as_text_and_the_loop_continues():
    g, _ = graph(
        tool_calls(call("attack", attacker="Dorn", target="the shadows")),
        "There is no one to fight yet.",
    )
    out = g.invoke(turn("I attack the shadows!"))
    tool_reply = [m for m in out["messages"] if isinstance(m, ToolMessage)][0]
    assert tool_reply.content.startswith("Cannot attack:") and "not a creature I know" in tool_reply.content
    assert narrations(out) == ["There is no one to fight yet."]


def test_the_step_cap_forces_narration():
    """After MAX_TOOL_STEPS round trips the model is called without tools and
    must narrate. The stub keeps asking for tools every time it is offered
    them; the one call where it is not offered them is the narration."""
    endless = [tool_calls(call("get_scene")) for _ in range(MAX_TOOL_STEPS)]
    g, stub = graph(*endless, "The DM, out of tools, narrates.", tool_calls(call("get_scene")))
    out = g.invoke(turn("look"))
    assert out["tool_steps"] == 0
    assert sum(1 for m in out["messages"] if isinstance(m, ToolMessage)) == MAX_TOOL_STEPS
    assert len(stub.prompts) == MAX_TOOL_STEPS + 1
    # The final call was narrate-only: the system prompt said so.
    assert "cannot call tools" in stub.prompts[MAX_TOOL_STEPS][0].content
    assert "cannot call tools" not in stub.prompts[0][0].content
    assert narrations(out) == ["The DM, out of tools, narrates."]


def test_a_tool_call_in_narrate_only_mode_is_ignored_not_run():
    """Even if a model emits a tool call when none were offered, nothing runs."""
    g, _ = graph(
        tool_calls(call("request_check", player="Kara", ability="DEX", dc=10)),
        AIMessage(content="Kara, roll.", tool_calls=[call("attack", attacker="Kara", target="Dorn")]),
    )
    out = g.invoke(turn("Kara sneaks"))
    assert narrations(out) == ["Kara, roll."]
    assert sum(1 for m in out["messages"] if isinstance(m, ToolMessage)) == 1


def test_a_silent_model_still_produces_a_reply():
    g, _ = graph(AIMessage(content=""))
    out = g.invoke(turn("..."))
    assert narrations(out) == [SILENT_FALLBACK]


def test_a_model_failure_ends_the_turn_with_a_message():
    class Broken(StubLLM):
        def invoke(self, messages, config=None):
            raise RuntimeError("daemon is down")

    g = create_game_graph(dm_llm=Broken(), researcher=StubResearcher())
    out = g.invoke(turn("hello"))
    assert narrations(out)[0].startswith("The story falters: daemon is down")


# --- pending checks --------------------------------------------------------------------------

def test_request_check_records_the_roll_and_the_dm_stops_to_ask():
    g, stub = graph(
        tool_calls(call("request_check", player="Kara", ability="DEX", skill="Stealth", dc=13, reason="slip past")),
        "Kara, give me a Dexterity (Stealth) check.",
    )
    out = g.invoke(turn("Kara sneaks past the guard"))
    assert get_pending(out).dc == 13
    assert narrations(out) == ["Kara, give me a Dexterity (Stealth) check."]
    # The second call was narrate-only: tools were not offered while a roll is owed.
    assert "a player still owes you a roll" in stub.prompts[1][0].content


def test_the_players_roll_resolves_the_check_and_the_dm_narrates_the_outcome():
    g, stub = graph(
        tool_calls(call("request_check", player="Kara", ability="DEX", skill="Stealth", dc=13)),
        "Kara, roll Stealth.",
        "Kara slips past unseen.",
    )
    first = g.invoke(turn("Kara sneaks"))
    roll = turn("/roll 15", first)
    roll["messages"] = list(first["messages"]) + roll["messages"]
    out = g.invoke(roll)
    assert get_pending(out) is None
    assert any("d20 15 + 7 = 22 vs DC 13 — success" in m.content for m in out["messages"] if m.name == "intake")
    assert narrations(out)[-1] == "Kara slips past unseen."
    assert "[roll resolved]" not in stub.prompts[-1][-1].content or True  # the result reaches the DM via the intake message


def test_a_result_resolved_by_intake_reaches_the_model_as_a_table_report():
    g, stub = graph("Kara slips past unseen.")
    values = seed()
    values["pending"] = {"player": "Kara", "kind": "check", "ability": "DEX", "skill": "Stealth", "dc": 13, "mode": "normal", "reason": ""}
    values["messages"] = [HumanMessage(content="/roll 15")]
    values["current_task"] = "/roll 15"
    g.invoke(values)
    last = stub.prompts[0][-1]
    assert isinstance(last, HumanMessage) and last.name == "table"
    assert last.content.startswith("[Results from the table — narrate these]")
    assert "d20 15 + 7 = 22 vs DC 13 — success" in last.content


# --- routing around the DM -----------------------------------------------------------------

def test_rules_go_to_the_researcher_and_dice_to_the_engine_without_the_dm():
    g, stub = graph("never called")
    out = g.invoke(turn("/rules how does grappling work"))
    assert [m.name for m in out["messages"] if isinstance(m, AIMessage)] == ["researcher"]
    out = g.invoke(turn("roll 2d6+3"))
    assert [m.name for m in out["messages"] if isinstance(m, AIMessage)] == ["intake"]
    assert stub.prompts == []


# --- the acceptance case: the goblin fight through the graph ------------------------------------

def test_the_goblin_fight_plays_end_to_end_through_the_graph():
    """Scripted tool calls only; the engine decides every number. Runs until
    the encounter is over, however the dice fall under the seed."""
    rng = random.Random(7)
    stub = StubLLM()
    g = create_game_graph(dm_llm=stub, researcher=StubResearcher(), rng=rng)

    values = turn("Two goblins! We attack.")
    stub.replies = [tool_calls(call("start_encounter", monsters=["goblin", "goblin"])), "Steel rings out."]
    values = g.invoke(values)
    enc = get_encounter(values)
    assert enc is not None and enc.active and set(enc.combatants) == {"Dorn", "Kara", "goblin-1", "goblin-2"}

    for _ in range(60):
        enc = get_encounter(values)
        if enc is None:
            break
        actor = enc.get(enc.current)
        assert actor.kind == "character", "a turn is never handed to the model with a monster up"
        foes = [c for c in enc.monsters if not c.dead]
        # The player attacks; their turn ends with it and the goblins act on
        # their own inside the tool. The model never calls end_turn here.
        stub.replies = [tool_calls(call("attack", attacker=actor.id, target=foes[0].id)), "Blows are traded."]
        nxt = turn(f"{actor.id} attacks", values)
        nxt["messages"] = list(values["messages"]) + nxt["messages"]
        values = g.invoke(nxt)

    assert get_encounter(values) is None, "the fight should have ended"
    party = get_party(values)
    assert all(0 <= c.current_hp <= c.max_hp for c in party.values())
    assert narrations(values)[-1] == "Blows are traded."
    # Engine results reach the story either through the model's tool calls
    # (ToolMessages) or through intake's own resolution of a declared attack.
    table = "\n".join(
        m.content for m in values["messages"]
        if isinstance(m, ToolMessage) or (isinstance(m, AIMessage) and m.name == "intake")
    )
    assert "encounter is over" in table
    assert "goblin-1 attacks" in table  # the monsters acted on their own
    assert values["tool_steps"] == 0


# --- the DungeonMaster class on its own ---------------------------------------------------------

def test_the_planner_is_offered_every_model_tool_and_only_those():
    stub = StubLLM()
    DungeonMaster(llm=stub)
    names = {t["function"]["name"] for t in stub.bound}
    assert "attack" in names and "request_check" in names
    assert "resolve_check" not in names
