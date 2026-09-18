"""`src/tools/` — every tool, offline, over a fake state; and the dispatcher's
promise that nothing ever raises.
"""

import random

import pytest

pytestmark = pytest.mark.integration  # state accessors import langgraph

pytest.importorskip("langgraph")

from langchain_core.documents import Document  # noqa: E402

from src.engine.character import Condition  # noqa: E402
from src.engine.pregens import pregen  # noqa: E402
from src.graph.game_state import (  # noqa: E402
    create_default_game_state,
    get_encounter,
    get_party,
    get_pending,
    put_party,
)
from src.tools import TOOLS, ToolResult, model_tools, run_tool  # noqa: E402
from src.tools import tools as t  # noqa: E402


class ScriptedRng:
    def __init__(self, *values):
        self.values = list(values)

    def randint(self, low, high):
        return self.values.pop(0)


def state_with(party=None, **overrides):
    s = dict(create_default_game_state())
    if party is None:
        party = {"Dorn": pregen("fighter", "p1", "Dorn"), "Kara": pregen("rogue", "p2", "Kara")}
    s["party"] = put_party(party)
    s.update(overrides)
    return s


def applied(state, result: ToolResult):
    """The state after a tool's update is written."""
    return {**state, **result.update}


def in_combat(rng=None):
    s = state_with()
    r = run_tool(s, "start_encounter", {"monsters": ["goblin", "goblin"]}, rng=rng or ScriptedRng(20, 15, 10, 5))
    assert r.ok, r.text
    return applied(s, r)


# --- read-only ------------------------------------------------------------------------

def test_get_scene_out_of_combat():
    text = run_tool(state_with(), "get_scene").text
    assert "Dorn (Human Fighter 1): AC 18, 13/13 HP" in text
    assert "Not in combat." in text


def test_get_scene_with_no_party():
    assert "No one has joined" in run_tool(state_with(party={}), "get_scene").text


def test_get_scene_in_combat_shows_the_sheet_and_the_pending_roll():
    s = in_combat()
    s = applied(s, run_tool(s, "request_check", {"player": "Kara", "ability": "DEX", "skill": "Stealth", "dc": 13}))
    text = run_tool(s, "get_scene").text
    assert "Round 1 — Dorn's turn." in text
    assert "Waiting on Kara for a Dexterity (Stealth) check (DC 13)." in text


def test_lookup_monster():
    text = run_tool({}, "lookup_monster", {"name": "goblin"}).text
    assert text.startswith("Goblin — Small humanoid, neutral evil. CR 0.25 (50 XP).")
    assert "AC 15, 7 HP" in text and "Scimitar (melee, +4 to hit, 1d6+2 slashing)" in text
    assert "Trait: Nimble Escape." in text


def test_lookup_monster_is_fuzzy_and_explains_misses():
    assert "Goblin —" in run_tool({}, "lookup_monster", {"name": "gobln"}).text
    miss = run_tool({}, "lookup_monster", {"name": "beholder"})
    assert not miss.ok and "No monster called 'beholder'" in miss.text


def test_lookup_rules_returns_labelled_passages_and_calls_no_model():
    fake = lambda q: [Document(page_content="Sneak Attack. Once per turn...", metadata={"source": "SRD 5.1", "category": "Class Features", "name": "Sneak Attack"})]
    text = run_tool({}, "lookup_rules", {"question": "how does sneak attack work"}, retriever=fake).text
    assert "[SRD 5.1, Class Features: Sneak Attack]" in text and "Once per turn" in text


def test_lookup_rules_with_nothing_found():
    assert "nothing close" in run_tool({}, "lookup_rules", {"question": "x"}, retriever=lambda q: []).text


# --- checks ------------------------------------------------------------------------------

def test_request_check_records_a_pending_roll_and_hides_the_dc():
    s = state_with()
    r = run_tool(s, "request_check", {"player": "kara", "ability": "DEX", "skill": "Stealth", "dc": 13, "reason": "slip past the guard"})
    assert r.ok and "Asked Kara for a Dexterity (Stealth) check — slip past the guard" in r.text
    assert "13" not in r.text
    pending = get_pending(applied(s, r))
    assert (pending.player, pending.dc, pending.skill.value) == ("Kara", 13, "Stealth")


def test_request_check_is_not_a_stand_in_for_an_attack_in_combat():
    """Measured: the 7B asks for a 'Strength check to hit'. It gets redirected."""
    r = run_tool(in_combat(), "request_check", {"player": "Dorn", "ability": "STR", "dc": 15, "reason": "to hit the goblin"})
    assert not r.ok and "Use attack(attacker='Dorn'" in r.text and r.update == {}
    # Out of combat, or for a non-attack reason, it is a normal check.
    assert run_tool(state_with(), "request_check", {"player": "Dorn", "ability": "STR", "dc": 15, "reason": "to hit the target dummy"}).ok
    assert run_tool(in_combat(), "request_check", {"player": "Dorn", "ability": "STR", "dc": 15, "reason": "to shove the door"}).ok


def test_request_check_for_an_unknown_player_names_the_party():
    r = run_tool(state_with(), "request_check", {"player": "Bob", "ability": "DEX", "dc": 10})
    assert not r.ok and "No party member called 'Bob'" in r.text and "Dorn, Kara" in r.text


def test_request_check_carries_condition_disadvantage():
    party = {"Kara": pregen("rogue", "p2", "Kara").model_copy(update={"conditions": [Condition.POISONED]})}
    s = state_with(party=party)
    pending = get_pending(applied(s, run_tool(s, "request_check", {"player": "Kara", "ability": "WIS", "skill": "Perception", "dc": 12})))
    assert pending.mode.value == "disadvantage"


def test_request_save_auto_fails_when_the_rules_say_so():
    party = {"Kara": pregen("rogue", "p2", "Kara").model_copy(update={"conditions": [Condition.PARALYZED]})}
    r = run_tool(state_with(party=party), "request_save", {"player": "Kara", "ability": "DEX", "dc": 15})
    assert r.ok and "automatically fails" in r.text and r.update == {}


def test_resolve_check_with_the_players_die():
    s = state_with()
    s = applied(s, run_tool(s, "request_check", {"player": "Kara", "ability": "DEX", "skill": "Stealth", "dc": 13}))
    r = run_tool(s, "resolve_check", {"d20": 5})
    assert "Kara: Dexterity (Stealth) check: d20 5 + 7 = 12 vs DC 13 — failure" in r.text
    assert get_pending(applied(s, r)) is None


def test_resolve_check_rolls_when_no_die_is_given():
    s = state_with()
    s = applied(s, run_tool(s, "request_save", {"player": "Dorn", "ability": "CON", "dc": 10, "reason": "the poison"}))
    r = run_tool(s, "resolve_check", {}, rng=ScriptedRng(9))
    assert "Dorn: Constitution save: d20 9 + 5 = 14 vs DC 10 — success (the poison)" in r.text


def test_resolve_check_with_nothing_pending():
    r = run_tool(state_with(), "resolve_check", {"d20": 10})
    assert not r.ok and "Nobody has been asked" in r.text


def test_resolve_check_is_not_offered_to_the_model():
    assert "resolve_check" not in [s.name for s in model_tools()]
    assert "resolve_check" in TOOLS


# --- combat -------------------------------------------------------------------------------

def test_start_encounter_numbers_monsters_and_rolls_initiative():
    s = in_combat()
    enc = get_encounter(s)
    assert set(enc.combatants) == {"Dorn", "Kara", "goblin-1", "goblin-2"}
    assert enc.order[0] == "Dorn"  # d20 20 + 2


def test_start_encounter_text_has_the_initiative_and_the_sheet():
    r = run_tool(state_with(), "start_encounter", {"monsters": ["goblin"]}, rng=ScriptedRng(20, 15, 10))
    assert "Roll for initiative!" in r.text and "Round 1 — Dorn's turn." in r.text


def test_start_encounter_refuses_a_second_fight_or_an_empty_party():
    s = in_combat()
    r = run_tool(s, "start_encounter", {"monsters": ["wolf"]})
    assert not r.ok and "already under way" in r.text
    r = run_tool(state_with(party={}), "start_encounter", {"monsters": ["wolf"]})
    assert not r.ok and "Nobody is in the party" in r.text


def test_start_encounter_with_an_unknown_monster():
    r = run_tool(state_with(), "start_encounter", {"monsters": ["goblin", "beholder"]})
    assert not r.ok and "No monster called 'beholder'" in r.text


def test_attack_resolves_and_updates_the_encounter():
    s = in_combat()
    r = run_tool(s, "attack", {"attacker": "Dorn", "target": "goblin-1"}, rng=ScriptedRng(12, 4))
    assert r.ok and "Dorn attacks goblin-1 with Longsword: d20 12 + 5 = 17 vs AC 15 — hit." in r.text
    assert "goblin-1 takes 7 slashing damage — 0/7 HP." in r.text and "goblin-1 dies." in r.text
    assert get_encounter(applied(s, r)).get("goblin-1").dead
    assert get_encounter(applied(s, r)).current == "Kara"  # Dorn's turn ended with the attack


def test_a_players_attack_ends_their_turn_and_the_monsters_act():
    """Dorn attacks; then goblin-1 and goblin-2 take their turns on their own;
    the turn comes back to Dorn (or the fight ends). The model never drives a
    monster."""
    s = in_combat()  # order: Dorn, Kara, goblin-1, goblin-2 (scripted initiative)
    r = run_tool(s, "attack", {"attacker": "Dorn", "target": "goblin-1"}, rng=random.Random(3))
    assert r.ok
    kinds = [e.kind for e in r.events]
    assert kinds[0] == "attack" and "turn_ended" in kinds
    after = get_encounter(applied(s, r))
    if after is not None:
        assert after.current == "Kara"  # the next player, not a goblin
        assert "It is now Kara's turn." in r.text
    goblin_attacks = [e for e in r.events if e.kind == "attack" and e.actor.startswith("goblin")]
    # Kara's turn comes before the goblins', so no goblin acted yet here.
    assert goblin_attacks == []


def test_after_the_last_player_the_monsters_all_act_before_returning():
    s = in_combat()
    s = applied(s, run_tool(s, "end_turn"))  # Dorn passes → Kara
    r = run_tool(s, "end_turn", rng=random.Random(5))  # Kara passes → goblins act → Dorn
    goblin_attacks = [e for e in r.events if e.kind == "attack" and e.actor.startswith("goblin")]
    after = get_encounter(applied(s, r))
    if after is not None:
        assert after.current == "Dorn" and after.round == 2
        assert len(goblin_attacks) == 2
        assert all(e.target in {"Dorn", "Kara"} for e in goblin_attacks)


def test_a_monster_that_wins_initiative_acts_before_the_players_hear_about_it():
    """Whichever seed, the encounter is never handed over with a monster up."""
    for seed in range(12):
        s = state_with()
        r = run_tool(s, "start_encounter", {"monsters": ["goblin", "wolf"]}, rng=random.Random(seed))
        assert r.ok, r.text
        after = get_encounter(applied(s, r))
        if after is None:
            continue  # the party went down before acting — lawful, if unlucky
        assert after.current in {"Dorn", "Kara"}
        first = after.order[0]
        if first not in {"Dorn", "Kara"}:
            assert any(e.kind == "attack" and e.actor == first for e in r.events)


def test_attack_out_of_turn_is_explained():
    r = run_tool(in_combat(), "attack", {"attacker": "Kara", "target": "goblin-1"}, rng=ScriptedRng(12, 4))
    assert not r.ok and r.text == "Cannot attack: It is Dorn's turn, not Kara's."


def test_attack_with_no_fight_starts_one_and_the_attacker_goes_first():
    """Whatever initiative says, the one who opened the fight strikes first —
    including when another party member would have out-rolled them."""
    for seed in range(8):
        s = state_with()
        r = run_tool(s, "attack", {"attacker": "Kara", "target": "goblin"}, rng=random.Random(seed))
        assert r.ok, r.text
        assert "Roll for initiative!" in r.text and "Kara attacks goblin-1 with Rapier" in r.text
        kinds = [e.kind for e in r.events]
        assert kinds.index("attack") > kinds.index("round_started")
        enc = get_encounter(applied(s, r))
        assert enc is None or enc.order[0] == "Kara"


def test_attack_with_no_fight_and_no_such_creature_is_explained():
    r = run_tool(state_with(), "attack", {"attacker": "Dorn", "target": "the shadows"})
    assert not r.ok and "is not a creature I know" in r.text and "start_encounter" in r.text


def test_attack_with_a_named_weapon_and_situational_advantage():
    s = in_combat()
    r = run_tool(s, "attack", {"attacker": "Dorn", "target": "goblin-2", "attack_name": "handaxe", "mode": "advantage"}, rng=ScriptedRng(3, 15, 2, 7))
    assert "with Handaxe" in r.text and "(advantage)" in r.text


def test_the_last_kill_ends_the_fight_and_syncs_the_party():
    s = in_combat()
    s = applied(s, run_tool(s, "apply_damage", {"target": "Dorn", "amount": 4}))
    s = applied(s, run_tool(s, "apply_damage", {"target": "goblin-2", "amount": 99}))
    r = run_tool(s, "attack", {"attacker": "Dorn", "target": "goblin-1"}, rng=ScriptedRng(20, 4, 4))
    assert "Victory!" in r.text and "The encounter is over (victory)." in r.text and "It is now" not in r.text
    after = applied(s, r)
    assert get_encounter(after) is None
    assert get_party(after)["Dorn"].current_hp == 9  # the fight's damage reached the sheet


def test_end_turn_advances_to_the_next_player():
    s = in_combat()
    r = run_tool(s, "end_turn")
    assert r.ok and r.text.endswith("It is now Kara's turn.")
    assert get_encounter(applied(s, r)).current == "Kara"


def test_end_turn_rolls_a_downed_characters_death_save_then_the_monsters_act():
    s = in_combat()
    s = applied(s, run_tool(s, "apply_damage", {"target": "Kara", "amount": 9}))
    r = run_tool(s, "end_turn", rng=random.Random(2))
    assert "death save" in r.text
    after = get_encounter(applied(s, r))
    assert after is None or after.current == "Dorn"


def test_end_encounter_by_fiat_syncs_and_clears():
    s = in_combat()
    r = run_tool(s, "end_encounter", {"outcome": "fled"})
    assert "The fight ends (fled)." in r.text and get_encounter(applied(s, r)) is None


# --- hp and conditions, in and out of combat ---------------------------------------------

def test_damage_out_of_combat_hits_the_sheet():
    s = state_with()
    r = run_tool(s, "apply_damage", {"target": "Dorn", "amount": 5, "damage_type": "fire"})
    assert "Dorn takes 5 fire damage — 8/13 HP." in r.text
    assert get_party(applied(s, r))["Dorn"].current_hp == 8


def test_damage_in_combat_hits_the_combatant_and_can_end_the_fight():
    s = in_combat()
    r = run_tool(s, "apply_damage", {"target": "goblin-1", "amount": 7})
    assert "goblin-1 dies." in r.text and get_encounter(applied(s, r)).get("goblin-1").dead


def test_heal_and_conditions_round_trip():
    s = state_with()
    s = applied(s, run_tool(s, "apply_damage", {"target": "Kara", "amount": 6}))
    r = run_tool(s, "heal", {"target": "Kara", "amount": 100})
    assert "regains 6 HP — 9/9." in r.text
    s = applied(s, run_tool(s, "apply_condition", {"target": "Kara", "condition": "prone"}))
    assert Condition.PRONE in get_party(s)["Kara"].conditions
    assert "already prone" in run_tool(s, "apply_condition", {"target": "Kara", "condition": "prone"}).text
    s = applied(s, run_tool(s, "remove_condition", {"target": "Kara", "condition": "prone"}))
    assert Condition.PRONE not in get_party(s)["Kara"].conditions


def test_conditions_on_monsters_in_combat():
    s = in_combat()
    r = run_tool(s, "apply_condition", {"target": "goblin-2", "condition": "restrained"})
    assert r.ok and get_encounter(applied(s, r)).get("goblin-2").has(Condition.RESTRAINED)


def test_rest_heals_the_party_and_is_refused_mid_fight():
    party = {"Dorn": pregen("fighter", "p1", "Dorn").model_copy(update={"current_hp": 2})}
    s = state_with(party=party)
    r = run_tool(s, "rest", {"kind": "short", "hit_dice": 1}, rng=ScriptedRng(10))
    assert "regain 11 HP" in r.text and get_party(applied(s, r))["Dorn"].current_hp == 13
    r = run_tool(s, "rest", {"kind": "long"})
    assert "wakes at full health" in r.text
    r = run_tool(in_combat(), "rest", {"kind": "long"})
    assert not r.ok and "middle of a fight" in r.text


# --- the dispatcher never raises ---------------------------------------------------------

def test_unknown_tool_lists_the_real_ones():
    r = run_tool({}, "cast_fireball", {})
    assert not r.ok and "no tool called 'cast_fireball'" in r.text and "attack" in r.text


@pytest.mark.parametrize("name", list(TOOLS))
@pytest.mark.parametrize("garbage", [
    {}, {"nonsense": 1}, {"player": 42}, {"target": None, "amount": "lots"},
    {"monsters": "goblin"}, {"dc": -5, "player": "Kara", "ability": "DEX"},
    {"ability": "LUCK", "player": "Kara", "dc": 10}, {"condition": "sleepy", "target": "Kara"},
])
def test_no_tool_ever_raises_on_bad_arguments(name, garbage):
    r = run_tool(state_with(), name, garbage, rng=random.Random(0))
    assert isinstance(r, ToolResult) and isinstance(r.text, str) and r.text


def test_validation_errors_name_the_field_and_what_the_tool_takes():
    r = run_tool(state_with(), "request_check", {"player": "Kara", "ability": "LUCK", "dc": 10})
    assert not r.ok and r.text.startswith("Cannot run request_check: ability:")
    assert "It takes: player, ability, skill, dc, reason." in r.text


def test_every_model_tool_has_a_description_and_a_schema():
    for spec in model_tools():
        assert spec.description and spec.args.model_json_schema()["type"] == "object"


def test_read_only_tools_write_nothing():
    for name, args in [("get_scene", {}), ("lookup_monster", {"name": "orc"})]:
        assert run_tool(state_with(), name, args).update == {}


def test_tools_never_mutate_the_state_they_are_given():
    s = state_with()
    before = repr(s)
    run_tool(s, "apply_damage", {"target": "Dorn", "amount": 5})
    run_tool(s, "start_encounter", {"monsters": ["goblin"]}, rng=random.Random(0))
    assert repr(s) == before
