"""Contract tests for `DungeonMaster`.

No model daemon: the narration client and the extractor are both replaced with
stubs. What is pinned here is what the agent does with a given model output —
context assembly, world-state merging, termination, and failure handling.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents.dungeon_master import CONTEXT_WINDOW, DungeonMaster

pytestmark = pytest.mark.integration  # constructing the agent imports the stack


class StubLLM:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def invoke(self, messages, *args, **kwargs):
        self.calls.append(messages)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def make_dm(narration="You push open the door.", scene=None):
    dm = DungeonMaster()
    dm.llm = StubLLM(
        narration if isinstance(narration, Exception)
        else AIMessage(content=narration)
    )
    dm.extractor = StubLLM(
        scene if scene is not None
        else {"location": "", "items_gained": [], "effects": []}
    )
    return dm


def state(task="I open the door", messages=None, world=None):
    return {
        "current_task": task,
        "messages": messages if messages is not None else [HumanMessage(content=task)],
        "game_state": world if world is not None else {},
    }


# --- the basic contract -----------------------------------------------------

def test_narration_is_returned_as_a_named_message():
    dm = make_dm("You push open the door.")
    command = dm.process_task(state())

    message = command.update["messages"][0]
    assert isinstance(message, AIMessage)
    assert message.name == "dungeon_master"
    assert message.content == "You push open the door."
    assert command.update["last_response"] == "You push open the door."


def test_the_node_terminates():
    """Every worker terminates since PR-04; handing back re-created #6."""
    dm = make_dm()
    assert dm.process_task(state()).goto == "__end__"


def test_the_annotation_matches_what_it_returns():
    """KNOWN_ISSUES #3 — LangGraph reads destinations off this annotation."""
    annotation = DungeonMaster.process_task.__annotations__["return"]
    assert "__end__" in str(annotation)
    assert "supervisor" not in str(annotation)


def test_the_interaction_is_logged(monkeypatch):
    dm = make_dm("You push open the door.")
    logged = []
    monkeypatch.setattr(dm, "_log_interaction",
                        lambda **kwargs: logged.append(kwargs))
    dm.process_task(state())
    assert logged and logged[0]["response"] == "You push open the door."


# --- context assembly -------------------------------------------------------

def test_dice_and_rules_output_stay_out_of_the_narrative_context():
    """Feeding mechanics back makes the DM narrate about the mechanics."""
    dm = make_dm()
    dm.process_task(state(messages=[
        HumanMessage(content="I attack the goblin"),
        AIMessage(content="🎲 Rolled 1d20: **17**", name="dice_roller"),
        AIMessage(content="Sneak attack, PHB p.90...", name="researcher"),
    ]))

    contents = [m.content for m in dm.llm.calls[0]]
    assert not any("Rolled" in c for c in contents)
    assert not any("PHB" in c for c in contents)
    assert any("I attack the goblin" in c for c in contents)


def test_its_own_prior_narration_is_kept():
    dm = make_dm()
    dm.process_task(state(messages=[
        HumanMessage(content="I open the door"),
        AIMessage(content="The door swings wide.", name="dungeon_master"),
        HumanMessage(content="I step through"),
    ]))
    assert any("The door swings wide." in m.content for m in dm.llm.calls[0])


def test_context_is_bounded():
    """Prompt-eval time is paid every turn; the campaign must not grow it."""
    dm = make_dm()
    dm.process_task(state(messages=[
        HumanMessage(content=f"turn {i}") for i in range(200)
    ]))
    assert len(dm.llm.calls[0]) <= CONTEXT_WINDOW + 1  # + the system message


def test_established_world_is_put_in_front_of_the_model():
    dm = make_dm()
    dm.process_task(state(world={
        "location": "the flooded crypt",
        "inventory": ["a rusted key"],
        "effects": ["torch burning low"],
    }))
    system = dm.llm.calls[0][0]
    assert isinstance(system, SystemMessage)
    assert "the flooded crypt" in system.content
    assert "a rusted key" in system.content
    assert "torch burning low" in system.content


def test_no_briefing_when_nothing_is_established():
    dm = make_dm()
    dm.process_task(state(world={}))
    assert "Established so far" not in dm.llm.calls[0][0].content


# --- world state ------------------------------------------------------------

def test_scene_updates_are_folded_into_game_state():
    dm = make_dm(scene={"location": "a collapsed stair",
                        "items_gained": ["a silver ring"],
                        "effects": ["ankle twisted"]})
    world = dm.process_task(state()).update["game_state"]
    assert world["location"] == "a collapsed stair"
    assert world["inventory"] == ["a silver ring"]
    assert world["effects"] == ["ankle twisted"]


def test_existing_world_state_survives_an_update():
    dm = make_dm(scene={"location": "the armoury", "items_gained": ["a shield"],
                        "effects": []})
    world = dm.process_task(state(world={
        "location": "the corridor", "inventory": ["a torch"], "npcs_met": ["Grix"],
    })).update["game_state"]

    assert world["location"] == "the armoury"      # replaced
    assert world["inventory"] == ["a torch", "a shield"]  # appended
    assert world["npcs_met"] == ["Grix"]           # untouched key preserved


def test_inventory_does_not_accumulate_duplicates():
    dm = make_dm(scene={"location": "", "items_gained": ["a torch"], "effects": []})
    world = dm.process_task(state(world={"inventory": ["a torch"]})).update["game_state"]
    assert world["inventory"] == ["a torch"]


def test_game_state_is_not_written_when_nothing_was_established():
    dm = make_dm(scene={"location": "", "items_gained": [], "effects": []})
    assert "game_state" not in dm.process_task(state()).update


def test_the_callers_world_dict_is_never_mutated():
    """`dict(state)` aliasing was a real bug here before (KNOWN_ISSUES #8)."""
    dm = make_dm(scene={"location": "the vault", "items_gained": ["a gem"],
                        "effects": []})
    original = {"location": "the hall", "inventory": ["a torch"]}
    dm.process_task(state(world=original))

    assert original == {"location": "the hall", "inventory": ["a torch"]}


# --- failure handling -------------------------------------------------------

def test_a_narration_failure_still_returns_a_command():
    """The node returning None is what #1 was; it must never regress to that."""
    dm = make_dm(RuntimeError("daemon is down"))
    command = dm.process_task(state())

    assert command.goto == "__end__"
    assert "daemon is down" in command.update["messages"][0].content


def test_an_empty_narration_is_treated_as_a_failure():
    dm = make_dm("   ")
    command = dm.process_task(state())
    assert command.goto == "__end__"
    assert command.update["messages"][0].content.strip()


def test_extraction_failure_does_not_lose_the_narration():
    """The story is the product; world state is a bonus."""
    dm = make_dm("You push open the door.", scene=RuntimeError("schema error"))
    command = dm.process_task(state())

    assert command.update["messages"][0].content == "You push open the door."
    assert "game_state" not in command.update
