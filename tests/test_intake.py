"""`src/graph/intake.py` — routing in code, dice in free text, the commands."""

import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from src.engine.pregens import pregen  # noqa: E402
from src.graph.game_state import create_default_game_state, get_party, get_pending, put_party, put_pending  # noqa: E402
from src.graph.intake import extract_dice_expression, intake, is_dice_request, parse_command, roll_text  # noqa: E402
from src.engine.checks import PendingCheck  # noqa: E402


class ScriptedRng:
    def __init__(self, *values):
        self.values = list(values)

    def randint(self, low, high):
        return self.values.pop(0)


def state(text, party=None, pending=None, speaker=None):
    s = dict(create_default_game_state())
    s["messages"] = [HumanMessage(content=text, name=speaker)]
    s["current_task"] = text
    if party:
        s["party"] = put_party(party)
    if pending:
        s["pending"] = put_pending(pending)
    return s


def reply(command):
    return command.update["messages"][0].content


# --- routing ------------------------------------------------------------------------------

def test_play_goes_to_the_dungeon_master_with_a_fresh_step_budget():
    c = intake(state("I push open the crypt door"))
    assert c.goto == "dungeon_master" and c.update == {"tool_steps": 0}


def test_rules_questions_go_to_the_researcher_stripped_of_the_command():
    c = intake(state("/rules how does grappling work"))
    assert c.goto == "researcher" and c.update == {"current_task": "how does grappling work"}


def test_an_empty_rules_command_asks_for_a_question():
    c = intake(state("/rules"))
    assert c.goto == "__end__" and "Ask something" in reply(c)


def test_questions_that_mention_dice_are_play_not_rolls():
    c = intake(state("what does d20 mean?"))
    assert c.goto == "dungeon_master"


def test_no_model_is_involved_anywhere():
    import inspect

    import src.graph.intake as module

    source = inspect.getsource(module)
    assert "create_llm" not in source and "invoke(" not in source


# --- dice in free text ---------------------------------------------------------------------

@pytest.mark.parametrize("text", ["2d6+3", "roll 1d20", "Roll 2d8 + 1d6 for damage", "d20 + 5", "rolling 1d20 with advantage"])
def test_plain_dice_requests_are_recognised(text):
    assert is_dice_request(text)


@pytest.mark.parametrize("text", ["my sword does 2d6 slashing, is that right", "how much is 1d8", "I attack the goblin", ""])
def test_dice_mentioned_in_passing_are_not_requests(text):
    assert not is_dice_request(text)


def test_extraction_reads_only_what_is_written():
    assert extract_dice_expression("roll 2d8 + 1d6 + 3 for damage") == ("2d8+1d6", 3)
    assert extract_dice_expression("roll a d20") == ("1d20", 0)
    assert extract_dice_expression("1d20-2") == ("1d20", -2)
    assert extract_dice_expression("nothing here") == (None, 0)


def test_subtracted_dice_are_refused():
    with pytest.raises(ValueError):
        extract_dice_expression("2d6-1d4")


def test_a_bare_roll_is_answered_by_the_engine_and_ends_the_turn():
    c = intake(state("roll 2d6+3 for damage"), rng=ScriptedRng(4, 5))
    assert c.goto == "__end__"
    assert reply(c) == "🎲 Rolled 2d6 + 3 for damage: **12** (2d6: [4, 5] = 9)"
    assert c.update["messages"][0].name == "intake"


def test_advantage_rolls_twice_and_keeps_the_higher():
    text = roll_text("roll 1d20 with advantage for stealth", ScriptedRng(3, 17))
    assert "with advantage" in text and "**17**" in text and "for stealth" in text
    text = roll_text("1d20 with disadvantage", ScriptedRng(3, 17))
    assert "**3**" in text


def test_unreadable_dice_get_a_hint_not_a_traceback():
    c = intake(state("roll 2d6-1d4"))
    assert c.goto == "__end__" and "could not read that as dice" in reply(c)


# --- /roll and pending checks -----------------------------------------------------------------

def party():
    return {"Kara": pregen("rogue", "p2", "Kara")}


def pending():
    return PendingCheck(player="Kara", kind="check", ability="DEX", skill="Stealth", dc=13)


def test_roll_with_a_pending_check_resolves_it_and_hands_the_dm_the_result():
    c = intake(state("/roll 5", party=party(), pending=pending()))
    assert c.goto == "dungeon_master"
    assert c.update["pending"] is None and c.update["tool_steps"] == 0
    assert reply(c) == "Kara: Dexterity (Stealth) check: d20 5 + 7 = 12 vs DC 13 — failure"
    assert c.update["current_task"].startswith("[roll resolved]")


def test_roll_with_no_die_rolls_for_the_player():
    c = intake(state("/roll", party=party(), pending=pending()), rng=ScriptedRng(10))
    assert "d20 10 + 7 = 17 vs DC 13 — success" in reply(c)


def test_a_bare_number_answers_a_pending_check():
    c = intake(state("18", party=party(), pending=pending()))
    assert c.goto == "dungeon_master" and "d20 18" in reply(c)


def test_play_while_a_roll_is_owed_gets_a_reminder_not_the_dm():
    c = intake(state("I charge the guard instead", party=party(), pending=pending()))
    assert c.goto == "__end__"
    assert "waiting on Kara for a Dexterity (Stealth) check" in reply(c) and "/roll" in reply(c)


def test_a_bare_number_with_nothing_pending_is_play():
    assert intake(state("18")).goto == "dungeon_master"


def test_an_out_of_range_die_is_refused():
    c = intake(state("/roll 25", party=party(), pending=pending()))
    assert c.goto == "__end__" and "1–20" in reply(c)


def test_roll_with_nothing_pending_needs_notation():
    c = intake(state("/roll"))
    assert "Nobody has asked you for a roll" in reply(c)
    c = intake(state("/roll 1d20+5"), rng=ScriptedRng(11))
    assert "**16**" in reply(c)


# --- /join, /party, /help -----------------------------------------------------------------------

def test_join_hands_out_a_pregen_owned_by_the_speaker():
    c = intake(state("/join rogue as Pip", speaker="discord:42"))
    party_after = get_party({"party": c.update["party"]})
    assert list(party_after) == ["Pip"]
    assert party_after["Pip"].player_id == "discord:42" and party_after["Pip"].character_class == "Rogue"
    assert "Pip joins the party" in reply(c)


def test_join_without_a_name_keeps_the_pregens_name():
    c = intake(state("/join fighter"))
    assert "Dorn Ironfist joins the party" in reply(c)


def test_join_refuses_a_duplicate_name_and_an_unknown_pregen():
    c = intake(state("/join rogue as Kara", party=party()))
    assert "already a Kara" in reply(c) and "party" not in c.update
    c = intake(state("/join paladin"))
    assert "No pregen called 'paladin'" in reply(c)


def test_join_with_no_argument_lists_the_choices():
    assert "rogue: Kara Swiftfoot" in reply(intake(state("/join")))


def test_party_shows_the_scene():
    text = reply(intake(state("/party", party=party())))
    assert "Kara (Halfling Rogue 1): AC 14, 9/9 HP" in text and "Not in combat." in text


def test_help_and_unknown_commands():
    assert "/join" in reply(intake(state("/help")))
    c = intake(state("/dance"))
    assert "Unknown command /dance" in reply(c)


def test_parse_command():
    assert parse_command("/rules  how does it work ") == ("rules", "how does it work")
    assert parse_command("/Roll") == ("roll", "")
    assert parse_command("I open the door") is None


def test_falls_back_to_the_last_human_message_when_current_task_is_empty():
    s = state("")
    s["messages"] = [HumanMessage(content="/help")]
    assert "Commands:" in reply(intake(s))


# --- attacks declared in play, during a fight ----------------------------------------------

def fight(text, monsters=("goblin",), speaker=None):
    """A state mid-fight with Dorn up, and the player's text as the turn."""
    import random

    from src.tools import run_tool

    base = state("start", party={"Dorn": pregen("fighter", "p1", "Dorn"), "Kara": pregen("rogue", "p2", "Kara")})
    started = run_tool(base, "start_encounter", {"monsters": list(monsters)}, rng=ScriptedRng(20, 15, *([1] * len(monsters))))
    s = {**base, **started.update}
    s["messages"] = [HumanMessage(content=text, name=speaker)]
    s["current_task"] = text
    return s


def test_a_declared_attack_on_the_only_monster_is_resolved_before_the_dm():
    c = intake(fight("I swing my longsword at the goblin!"), rng=ScriptedRng(12, 4, 9, 3, 8, 2))
    assert c.goto == "dungeon_master"
    text = reply(c)
    assert text.startswith("Dorn attacks goblin-1 with Longsword: d20 12 + 5 = 17 vs AC 15 — hit.")
    assert "goblin-1 takes 7 slashing damage — 0/7 HP." in text and "goblin-1 dies." in text
    assert c.update["messages"][0].name == "intake"
    assert c.update["tool_steps"] == 0 and "encounter" in c.update


def test_the_named_weapon_is_used():
    c = intake(fight("I throw my handaxe at the goblin"), rng=ScriptedRng(12, 4, 9, 3, 8, 2))
    assert "with Handaxe" in reply(c)


def test_it_refers_to_the_only_monster_standing():
    c = intake(fight("I attack it"), rng=ScriptedRng(12, 4, 9, 3, 8, 2))
    assert "Dorn attacks goblin-1" in reply(c)


def test_two_goblins_and_the_goblin_is_ambiguous_so_the_dm_decides():
    c = intake(fight("I attack the goblin", monsters=("goblin", "goblin")), rng=ScriptedRng(12, 4))
    assert c.goto == "dungeon_master" and "messages" not in c.update


def test_naming_the_id_disambiguates():
    c = intake(fight("I attack goblin-2", monsters=("goblin", "goblin")), rng=ScriptedRng(12, 4, 9, 3, 8, 2, 7, 5))
    assert "Dorn attacks goblin-2" in reply(c)


def test_a_negated_attack_is_not_an_attack():
    c = intake(fight("I don't attack the goblin, I talk to it"))
    assert c.goto == "dungeon_master" and "messages" not in c.update


def test_a_question_about_attacking_is_not_an_attack():
    c = intake(fight("can I attack the goblin from here?"))
    assert c.goto == "dungeon_master" and "messages" not in c.update


def test_no_attack_verb_means_play():
    c = intake(fight("I back away from the goblin slowly"))
    assert c.goto == "dungeon_master" and "messages" not in c.update


def test_out_of_combat_an_attack_verb_is_play_for_the_dm_to_judge():
    """Measured: auto-starting fights from text began a second fight against a
    goblin the party had just killed. Out of a fight, the DM decides."""
    for text in ("I attack the goblin!", "I charge the adult red dragon", "I attack the darkness"):
        c = intake(state(text, party={"Dorn": pregen("fighter", "p1", "Dorn")}))
        assert c.goto == "dungeon_master" and "messages" not in c.update
