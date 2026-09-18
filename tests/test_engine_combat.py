"""`src/engine/combat.py` — one rule per test, and one whole fight.

Random sources are scripted (`ScriptedRng`) so each assertion is about the
rule; the end-to-end fight uses `random.Random(seed)` and proves it replays.
"""

import random

import pytest

pytest.importorskip("pydantic")

from src.engine.character import Ability, Character, Condition  # noqa: E402
from src.engine.checks import RollMode  # noqa: E402
from src.engine.combat import (  # noqa: E402
    Encounter,
    Event,
    RulesError,
    add_condition,
    apply_damage,
    attack,
    attack_mode,
    can_act,
    check_mode,
    damage_combatant,
    damage_multiplier,
    death_save,
    end_encounter,
    end_turn,
    grant_temp_hp,
    heal,
    heal_combatant,
    long_rest,
    remove_condition,
    save_mode,
    short_rest,
    stabilize,
    start_encounter,
    sync_party,
)
from src.engine.combatant import Combatant  # noqa: E402
from src.engine.pregens import pregen  # noqa: E402
from src.srd.bestiary import summon, summon_group  # noqa: E402


class ScriptedRng:
    def __init__(self, *values):
        self.values = list(values)

    def randint(self, low, high):
        return self.values.pop(0)


def fighter() -> Character:
    return pregen("fighter", "p1", "Dorn")


def rogue() -> Character:
    return pregen("rogue", "p2", "Kara")


def kinds(events) -> list:
    return [e.kind for e in events]


# --- condition effects --------------------------------------------------------

def test_incapacitating_conditions_stop_actions():
    g = summon("goblin")
    assert can_act(g)
    for c in (Condition.PARALYZED, Condition.STUNNED, Condition.UNCONSCIOUS, Condition.INCAPACITATED, Condition.PETRIFIED):
        assert not can_act(g.model_copy(update={"conditions": [c]})), c
    assert can_act(g.model_copy(update={"conditions": [Condition.PRONE]}))
    assert not can_act(g.model_copy(update={"dead": True}))


@pytest.mark.parametrize("condition", [Condition.BLINDED, Condition.POISONED, Condition.PRONE, Condition.RESTRAINED, Condition.FRIGHTENED])
def test_an_impaired_attacker_has_disadvantage(condition):
    a = summon("goblin").model_copy(update={"conditions": [condition]})
    assert attack_mode(a, summon("orc"), ranged=False) == RollMode.DISADVANTAGE


@pytest.mark.parametrize("condition", [Condition.BLINDED, Condition.RESTRAINED, Condition.PARALYZED, Condition.STUNNED, Condition.UNCONSCIOUS, Condition.PETRIFIED])
def test_a_helpless_target_grants_advantage(condition):
    t = summon("orc").model_copy(update={"conditions": [condition]})
    assert attack_mode(summon("goblin"), t, ranged=False) == RollMode.ADVANTAGE


def test_a_prone_target_is_easy_up_close_and_hard_at_range():
    t = summon("orc").model_copy(update={"conditions": [Condition.PRONE]})
    assert attack_mode(summon("goblin"), t, ranged=False) == RollMode.ADVANTAGE
    assert attack_mode(summon("goblin"), t, ranged=True) == RollMode.DISADVANTAGE


def test_advantage_and_disadvantage_cancel():
    a = summon("goblin").model_copy(update={"conditions": [Condition.POISONED]})
    t = summon("orc").model_copy(update={"conditions": [Condition.RESTRAINED]})
    assert attack_mode(a, t, ranged=False) == RollMode.NORMAL


def test_invisibility_cuts_both_ways():
    g, o = summon("goblin"), summon("orc")
    assert attack_mode(g.model_copy(update={"conditions": [Condition.INVISIBLE]}), o, False) == RollMode.ADVANTAGE
    assert attack_mode(g, o.model_copy(update={"conditions": [Condition.INVISIBLE]}), False) == RollMode.DISADVANTAGE


def test_poisoned_and_frightened_hurt_checks():
    assert check_mode(summon("goblin")) == RollMode.NORMAL
    assert check_mode(summon("goblin").model_copy(update={"conditions": [Condition.POISONED]})) == RollMode.DISADVANTAGE


def test_save_modes():
    g = summon("goblin")
    assert save_mode(g, Ability.DEX) == RollMode.NORMAL
    assert save_mode(g.model_copy(update={"conditions": [Condition.RESTRAINED]}), Ability.DEX) == RollMode.DISADVANTAGE
    assert save_mode(g.model_copy(update={"conditions": [Condition.RESTRAINED]}), Ability.WIS) == RollMode.NORMAL
    assert save_mode(g.model_copy(update={"conditions": [Condition.PARALYZED]}), Ability.STR) is None
    assert save_mode(g.model_copy(update={"conditions": [Condition.PARALYZED]}), Ability.CON) == RollMode.NORMAL


# --- damage -------------------------------------------------------------------

def test_damage_reduces_hp_and_says_so():
    g, events = apply_damage(summon("goblin"), 3, "slashing")
    assert g.current_hp == 4
    assert kinds(events) == ["damage"]
    assert "takes 3 slashing damage" in events[0].text and "4/7 HP" in events[0].text


def test_temporary_hp_absorb_first():
    g = summon("goblin").model_copy(update={"temp_hp": 2})
    g, events = apply_damage(g, 3)
    assert (g.temp_hp, g.current_hp) == (0, 6)
    assert "2 absorbed by temporary HP" in events[0].text


def test_temporary_hp_do_not_stack():
    g, _ = grant_temp_hp(summon("goblin"), 5)
    g, events = grant_temp_hp(g, 3)
    assert g.temp_hp == 5 and "would not be more" in events[0].text


def test_resistance_halves_rounded_down():
    assert damage_multiplier(summon("adult red dragon"), "fire") == 0.0
    skeleton = summon("skeleton")  # vulnerable to bludgeoning
    assert damage_multiplier(skeleton, "bludgeoning") == 2.0
    resistant = summon("goblin").model_copy(update={"damage_resistances": ["slashing"]})
    hurt, events = apply_damage(resistant, 5, "slashing")
    assert hurt.current_hp == 7 - 2 and "resisted, halved" in events[0].text


def test_immunity_zeroes_damage():
    dragon = summon("adult red dragon")
    same, events = apply_damage(dragon, 40, "fire")
    assert same.current_hp == dragon.current_hp and "immune" in events[0].text


def test_vulnerability_doubles():
    skeleton = summon("skeleton")
    hurt, _ = apply_damage(skeleton, 3, "bludgeoning")
    assert hurt.current_hp == skeleton.max_hp - 6


def test_a_monster_dies_at_zero():
    g, events = apply_damage(summon("goblin"), 7)
    assert g.dead and g.current_hp == 0 and not g.is_alive
    assert kinds(events) == ["damage", "died"]


def test_a_character_drops_unconscious_at_zero():
    c = Combatant.from_character(fighter())
    down, events = apply_damage(c, 13)
    assert not down.dead and down.current_hp == 0
    assert Condition.UNCONSCIOUS in down.conditions and not down.is_conscious
    assert kinds(events) == ["damage", "dropped"]
    assert (down.death_successes, down.death_failures, down.stable) == (0, 0, False)


def test_massive_damage_kills_a_character_outright():
    c = Combatant.from_character(fighter())  # 13 HP
    dead, events = apply_damage(c, 13 + 13)
    assert dead.dead and kinds(events) == ["damage", "died"] and "instantly" in events[1].text
    alive, _ = apply_damage(c, 13 + 12)
    assert not alive.dead


def test_damage_while_down_is_a_failed_death_save_two_on_a_crit():
    c = Combatant.from_character(fighter())
    down, _ = apply_damage(c, 13)
    down, events = apply_damage(down, 1)
    assert down.death_failures == 1 and kinds(events) == ["damage", "death_save"]
    down, events = apply_damage(down, 1, critical=True)
    assert down.death_failures == 3 and down.dead and kinds(events)[-1] == "died"


def test_the_dead_take_no_more_damage():
    g, _ = apply_damage(summon("goblin"), 99)
    with pytest.raises(RulesError):
        apply_damage(g, 1)


def test_a_plain_character_sheet_can_be_damaged_too():
    """Out of combat there is no Combatant; the sheet takes the hit directly."""
    c, events = apply_damage(fighter(), 5, "fire")
    assert c.current_hp == 8 and kinds(events) == ["damage"]
    c, events = apply_damage(c, 8)
    assert c.current_hp == 0 and Condition.UNCONSCIOUS in c.conditions and not c.dead


# --- healing ------------------------------------------------------------------

def test_healing_is_capped_at_max():
    g, _ = apply_damage(summon("goblin"), 4)
    g, events = heal(g, 10)
    assert g.current_hp == 7 and "regains 4 HP" in events[0].text


def test_healing_from_zero_restores_consciousness():
    c = Combatant.from_character(fighter())
    down, _ = apply_damage(c, 13)
    down, _ = apply_damage(down, 1)
    up, events = heal(down, 1)
    assert up.current_hp == 1 and up.is_conscious
    assert Condition.UNCONSCIOUS not in up.conditions
    assert (up.death_successes, up.death_failures) == (0, 0)
    assert "regains consciousness" in events[0].text


def test_the_dead_cannot_be_healed():
    g, _ = apply_damage(summon("goblin"), 99)
    with pytest.raises(RulesError):
        heal(g, 5)


# --- death saves --------------------------------------------------------------

def downed() -> Combatant:
    return apply_damage(Combatant.from_character(fighter()), 13)[0]


def test_ten_or_more_is_a_success():
    c, events = death_save(downed(), ScriptedRng(10))
    assert c.death_successes == 1 and "succeeds" in events[0].text


def test_nine_or_less_is_a_failure():
    c, _ = death_save(downed(), ScriptedRng(9))
    assert c.death_failures == 1


def test_a_natural_one_is_two_failures():
    c, _ = death_save(downed(), ScriptedRng(1))
    assert c.death_failures == 2


def test_a_natural_twenty_gets_up_with_one_hp():
    c, events = death_save(downed(), ScriptedRng(20))
    assert c.current_hp == 1 and c.is_conscious and "natural 20" in events[0].text


def test_three_successes_stabilise():
    c = downed()
    for _ in range(3):
        c, events = death_save(c, ScriptedRng(15))
    assert c.stable and kinds(events)[-1] == "stabilized"
    with pytest.raises(RulesError):
        death_save(c, ScriptedRng(15))  # stable: no more rolls


def test_three_failures_kill():
    c = downed()
    for _ in range(3):
        c, events = death_save(c, ScriptedRng(5))
    assert c.dead and kinds(events)[-1] == "died"


def test_only_the_dying_roll_death_saves():
    with pytest.raises(RulesError):
        death_save(Combatant.from_character(fighter()), ScriptedRng(10))


def test_stabilize_by_hand():
    c, events = stabilize(downed())
    assert c.stable and kinds(events) == ["stabilized"]


# --- conditions ---------------------------------------------------------------

def test_conditions_are_added_and_removed_with_events():
    g, events = add_condition(summon("goblin"), Condition.PRONE)
    assert g.has(Condition.PRONE) and kinds(events) == ["condition_added"]
    same, events = add_condition(g, Condition.PRONE)
    assert same == g and events == []
    g, events = remove_condition(g, "prone")
    assert not g.has(Condition.PRONE) and kinds(events) == ["condition_removed"]


def test_condition_immunities_are_honoured():
    skeleton = summon("skeleton")  # immune to poisoned, exhaustion
    same, events = add_condition(skeleton, Condition.POISONED)
    assert not same.has(Condition.POISONED) and kinds(events) == ["condition_immune"]


# --- the encounter ------------------------------------------------------------

def test_initiative_orders_highest_first_with_dex_then_name_for_ties():
    # d20 rolls: Dorn (+2), Kara (+3), goblin-1 (+2), goblin-2 (+2)
    enc, events = start_encounter([fighter(), rogue()], summon_group("goblin", 2), rng=ScriptedRng(10, 15, 16, 10))
    assert enc.initiative == {"Dorn": 12, "Kara": 18, "goblin-1": 18, "goblin-2": 12}
    assert enc.order == ["Kara", "goblin-1", "Dorn", "goblin-2"]  # Kara wins the tie on DEX; Dorn on name
    assert enc.current == "Kara" and enc.round == 1
    assert kinds(events) == ["encounter_started", "initiative", "initiative", "initiative", "initiative", "round_started", "turn_started"]


def test_party_members_become_character_combatants():
    enc, _ = start_encounter([fighter()], summon_group("goblin", 1), rng=random.Random(0))
    dorn = enc.get("Dorn")
    assert dorn.kind == "character" and dorn.player_id == "p1"
    assert dorn.attack("longsword").attack_bonus == 5
    assert dorn.attack("longsword").damage[0].notation == "1d8+3"
    assert dorn.save_modifier(Ability.STR) == 5 and dorn.skill_modifier("Athletics") == 5


def test_duplicate_names_are_refused():
    with pytest.raises(RulesError):
        start_encounter([fighter(), fighter()], [], rng=random.Random(0))


def test_lookup_by_id_or_name_is_forgiving():
    enc, _ = start_encounter([fighter()], summon_group("goblin", 1), rng=random.Random(0))
    assert enc.get("dorn").name == "Dorn"
    assert enc.get("Goblin").id == "goblin-1"
    with pytest.raises(RulesError) as info:
        enc.get("dragon")
    assert "Present: " in str(info.value)


def test_attacks_only_on_your_turn():
    enc, _ = start_encounter([fighter()], summon_group("goblin", 1), rng=ScriptedRng(20, 1))  # Dorn first
    with pytest.raises(RulesError) as info:
        attack(enc, "goblin-1", "Dorn", rng=ScriptedRng(15, 3))
    assert "Dorn's turn" in str(info.value)
    enc2, events = attack(enc, "goblin-1", "Dorn", rng=ScriptedRng(15, 3), enforce_turn=False)
    assert kinds(events)[0] == "attack"


def test_a_hit_rolls_every_damage_component_and_applies_it():
    enc, _ = start_encounter([fighter()], summon_group("goblin", 1), rng=ScriptedRng(20, 1))
    enc, events = attack(enc, "Dorn", "goblin-1", rng=ScriptedRng(12, 4))  # 12 + 5 = 17 vs AC 15; 1d8 → 4, +3
    assert kinds(events) == ["attack", "damage_rolled", "damage", "died", "encounter_ended"]
    assert "= 17" in events[0].text and "hit" in events[0].text
    assert "= 7 slashing" in events[1].text
    assert enc.get("goblin-1").dead and not enc.active and enc.outcome == "victory"


def test_a_miss_rolls_no_damage():
    enc, _ = start_encounter([fighter()], summon_group("goblin", 1), rng=ScriptedRng(20, 1))
    enc, events = attack(enc, "Dorn", "goblin-1", rng=ScriptedRng(5))
    assert kinds(events) == ["attack"] and "miss" in events[0].text
    assert enc.get("goblin-1").current_hp == 7


def test_a_natural_one_misses_and_a_twenty_crits():
    enc, _ = start_encounter([fighter()], summon_group("ogre", 1), rng=ScriptedRng(20, 1))
    _, events = attack(enc, "Dorn", "ogre-1", rng=ScriptedRng(1))
    assert "miss" in events[0].text
    _, events = attack(enc, "Dorn", "ogre-1", rng=ScriptedRng(20, 8, 8))
    assert "critical hit" in events[0].text and "2d8" in events[1].text and "= 19 slashing" in events[1].text


def test_a_melee_hit_on_a_paralyzed_target_is_a_critical():
    enc, _ = start_encounter([fighter()], summon_group("ogre", 1), rng=ScriptedRng(20, 1))
    ogre, _ = add_condition(enc.get("ogre-1"), Condition.PARALYZED)
    enc = enc.model_copy(update={"combatants": {**enc.combatants, "ogre-1": ogre}})
    _, events = attack(enc, "Dorn", "ogre-1", rng=ScriptedRng(3, 15, 6, 6))  # advantage: [3, 15] → 15
    assert "(advantage)" in events[0].text and "critical hit" in events[0].text


def test_an_incapacitated_attacker_is_refused():
    enc, _ = start_encounter([fighter()], summon_group("goblin", 1), rng=ScriptedRng(20, 1))
    dorn, _ = add_condition(enc.get("Dorn"), Condition.STUNNED)
    enc = enc.model_copy(update={"combatants": {**enc.combatants, "Dorn": dorn}})
    with pytest.raises(RulesError) as info:
        attack(enc, "Dorn", "goblin-1", rng=ScriptedRng(15))
    assert "stunned" in str(info.value)


def test_you_cannot_attack_the_dead_or_yourself():
    enc, _ = start_encounter([fighter()], summon_group("goblin", 1), rng=ScriptedRng(20, 1))
    with pytest.raises(RulesError):
        attack(enc, "Dorn", "Dorn", rng=ScriptedRng(15))
    enc, _ = damage_combatant(enc, "goblin-1", 99)
    with pytest.raises(RulesError):
        attack(enc, "Dorn", "goblin-1", rng=ScriptedRng(15), enforce_turn=False)


def test_choosing_a_named_attack():
    enc, _ = start_encounter([rogue()], summon_group("goblin", 1), rng=ScriptedRng(20, 1))
    _, events = attack(enc, "Kara", "goblin-1", attack_name="shortbow", rng=ScriptedRng(15, 4))
    assert "with Shortbow" in events[0].text and "piercing" in events[1].text


def test_end_turn_advances_wraps_and_skips_the_dead():
    enc, _ = start_encounter([fighter()], summon_group("goblin", 2), rng=ScriptedRng(20, 10, 5))  # Dorn, g1, g2
    enc, _ = damage_combatant(enc, "goblin-1", 99)
    enc, events = end_turn(enc)
    assert enc.current == "goblin-2" and kinds(events) == ["turn_ended", "turn_started"]
    enc, events = end_turn(enc)
    assert enc.current == "Dorn" and enc.round == 2 and "round_started" in kinds(events)


def test_a_downed_character_rolls_a_death_save_as_their_turn_begins_and_is_skipped():
    enc, _ = start_encounter([fighter(), rogue()], summon_group("goblin", 1), rng=ScriptedRng(20, 10, 5))  # Dorn, Kara, g1
    enc, _ = damage_combatant(enc, "Kara", 9)  # Kara down
    enc, events = end_turn(enc, rng=ScriptedRng(12))  # Kara's turn: save (success), then skipped to the goblin
    assert enc.current == "goblin-1"
    assert "death_save" in kinds(events) and enc.get("Kara").death_successes == 1


def test_the_party_going_down_ends_the_fight_in_defeat():
    enc, _ = start_encounter([fighter()], summon_group("goblin", 1), rng=ScriptedRng(20, 10))
    enc, events = damage_combatant(enc, "Dorn", 13)
    assert kinds(events)[-1] == "encounter_ended" and enc.outcome == "defeat" and not enc.active
    with pytest.raises(RulesError):
        end_turn(enc)


def test_dying_out_on_your_own_turn_ends_the_fight():
    enc, _ = start_encounter([fighter()], summon_group("goblin", 1), rng=ScriptedRng(1, 20))  # goblin first
    enc, _ = damage_combatant(enc, "Dorn", 13)
    enc = enc.model_copy(update={"active": True, "outcome": None})  # keep it going for the test
    enc, events = end_turn(enc, rng=ScriptedRng(1, 1))  # Dorn: two failures... then needs a third
    assert enc.get("Dorn").death_failures == 2


def test_heal_and_temp_hp_inside_an_encounter():
    enc, _ = start_encounter([fighter()], summon_group("goblin", 1), rng=ScriptedRng(20, 10))
    enc, _ = damage_combatant(enc, "Dorn", 5)
    enc, events = heal_combatant(enc, "Dorn", 3)
    assert enc.get("Dorn").current_hp == 11 and kinds(events) == ["healed"]


def test_end_encounter_by_fiat():
    enc, _ = start_encounter([fighter()], summon_group("goblin", 1), rng=ScriptedRng(20, 10))
    enc, events = end_encounter(enc, "fled")
    assert not enc.active and enc.outcome == "fled" and kinds(events) == ["encounter_ended"]
    assert end_encounter(enc) == (enc, [])


def test_sync_party_writes_the_fight_back_to_the_sheets():
    party = {"Dorn": fighter(), "Kara": rogue()}
    enc, _ = start_encounter(list(party.values()), summon_group("goblin", 1), rng=random.Random(0))
    enc, _ = damage_combatant(enc, "Dorn", 5)
    enc, _ = damage_combatant(enc, "Kara", 9)
    synced = sync_party(party, enc)
    assert synced["Dorn"].current_hp == 8
    assert synced["Kara"].current_hp == 0 and Condition.UNCONSCIOUS in synced["Kara"].conditions
    assert party["Dorn"].current_hp == 13  # the originals are untouched


def test_the_sheet_reads_like_a_scene():
    enc, _ = start_encounter([fighter()], summon_group("goblin", 2), rng=ScriptedRng(20, 10, 5))
    enc, _ = damage_combatant(enc, "goblin-1", 99)
    text = enc.sheet()
    assert text.startswith("Round 1 — Dorn's turn.")
    assert "Order: Dorn (22), goblin-1 (12), goblin-2 (7)" in text
    assert "✝ goblin-1 (Goblin): AC 15, 0/7 HP, dead" in text
    assert "Dorn: AC 18, 13/13 HP" in text


def test_encounter_and_events_round_trip_through_json():
    enc, events = start_encounter([fighter()], summon_group("goblin", 2), rng=random.Random(3))
    assert Encounter.model_validate_json(enc.model_dump_json()) == enc
    assert all(Event.model_validate_json(e.model_dump_json()) == e for e in events)


# --- rests --------------------------------------------------------------------

def test_a_short_rest_spends_hit_dice():
    c = fighter().model_copy(update={"current_hp": 3})
    rested, events = short_rest(c, hit_dice=1, rng=ScriptedRng(6))  # d10 6 + CON 3
    assert rested.current_hp == 12 and rested.hit_dice_remaining == 0
    assert kinds(events) == ["rest"] and "regain 9 HP" in events[0].text
    with pytest.raises(RulesError):
        short_rest(rested, hit_dice=1, rng=ScriptedRng(6))


def test_a_short_rest_never_heals_negative():
    weak = fighter().model_copy(update={"current_hp": 5, "abilities": fighter().abilities.model_copy(update={"constitution": 3})})
    rested, _ = short_rest(weak, 1, rng=ScriptedRng(1))
    assert rested.current_hp == 5


def test_a_long_rest_restores_everything_and_half_the_dice():
    c = fighter().model_copy(update={"current_hp": 1, "temp_hp": 4, "hit_dice_remaining": 0, "conditions": [Condition.UNCONSCIOUS]})
    rested, events = long_rest(c)
    assert rested.current_hp == 13 and rested.temp_hp == 0 and rested.hit_dice_remaining == 1
    assert Condition.UNCONSCIOUS not in rested.conditions and kinds(events) == ["rest"]


def test_the_dead_do_not_rest():
    with pytest.raises(RulesError):
        long_rest(fighter().model_copy(update={"dead": True}))


# --- the whole fight, deterministic ------------------------------------------

def play_out(seed: int):
    rng = random.Random(seed)
    party = {"Dorn": fighter(), "Kara": rogue()}
    enc, log = start_encounter(list(party.values()), summon_group("goblin", 2), rng=rng)
    for _ in range(200):
        if not enc.active:
            break
        actor = enc.get(enc.current)
        foes = [c for c in (enc.monsters if actor.kind == "character" else enc.characters) if not c.dead and c.current_hp > 0]
        if foes and can_act(actor):
            enc, events = attack(enc, actor.id, foes[0].id, rng=rng)
            log += events
        if enc.active:
            enc, events = end_turn(enc, rng=rng)
            log += events
    return enc, log, sync_party(party, enc)


def test_the_goblin_fight_runs_to_completion_and_replays_exactly():
    enc, log, party = play_out(7)
    assert not enc.active and enc.outcome in {"victory", "defeat"}
    assert kinds(log)[0] == "encounter_started" and kinds(log)[-1] == "encounter_ended"
    assert {"attack", "damage_rolled", "damage", "turn_ended", "turn_started"} <= set(kinds(log))
    assert all(isinstance(e.text, str) and e.text for e in log)
    again_enc, again_log, again_party = play_out(7)
    assert (again_enc, [e.text for e in again_log], again_party) == (enc, [e.text for e in log], party)
    different_enc, _, _ = play_out(8)
    assert different_enc != enc


def test_over_many_seeds_the_fight_always_ends_lawfully():
    for seed in range(20):
        enc, log, party = play_out(seed)
        assert not enc.active
        if enc.outcome == "victory":
            assert all(m.dead for m in enc.monsters)
            assert any(not c.dead and c.current_hp > 0 for c in enc.characters)
        else:
            assert all(c.dead or c.current_hp == 0 for c in enc.characters)
        for c in enc.combatants.values():
            assert 0 <= c.current_hp <= c.max_hp
