"""`src/engine/checks.py` — d20 resolution, one rule per test.

The random source is a scripted stand-in, so every assertion is about the
rule, not about luck. `ScriptedRng.randint` hands out the next number in its
list regardless of the range asked for.
"""

import random

import pytest

pytest.importorskip("pydantic")

from src.engine.character import Abilities, Ability, Character, Proficiency, Skill, Weapon  # noqa: E402
from src.engine.checks import (  # noqa: E402
    AttackResult,
    CheckResult,
    RollMode,
    ability_check,
    attack_roll,
    resolve_mode,
    roll_d20,
    roll_damage,
    saving_throw,
)


class ScriptedRng:
    def __init__(self, *values):
        self.values = list(values)
        self.asked = []

    def randint(self, low, high):
        self.asked.append((low, high))
        return self.values.pop(0)


ROGUE = Character(
    player_id="p", name="Kara", race="Halfling", character_class="Rogue",
    abilities=Abilities(strength=8, dexterity=17, constitution=13, intelligence=14, wisdom=12, charisma=11),
    skills={Skill.STEALTH: Proficiency.EXPERTISE, Skill.PERCEPTION: Proficiency.PROFICIENT},
    saves=[Ability.DEX, Ability.INT],
    armor_class=14, max_hp=9, hit_die=8,
)
RAPIER = Weapon(name="Rapier", damage_dice="1d8", damage_type="piercing", finesse=True)
LONGSWORD = Weapon(name="Longsword", damage_dice="1d8", damage_type="slashing", versatile_dice="1d10")


# --- advantage and disadvantage --------------------------------------------

def test_modes_cancel_when_both_are_present():
    assert resolve_mode(RollMode.ADVANTAGE, RollMode.DISADVANTAGE) == RollMode.NORMAL
    assert resolve_mode(RollMode.ADVANTAGE, RollMode.ADVANTAGE, RollMode.DISADVANTAGE) == RollMode.NORMAL


def test_a_single_mode_stands():
    assert resolve_mode(RollMode.ADVANTAGE) == RollMode.ADVANTAGE
    assert resolve_mode(RollMode.NORMAL, RollMode.DISADVANTAGE) == RollMode.DISADVANTAGE
    assert resolve_mode() == RollMode.NORMAL


def test_normal_rolls_one_die():
    kept, rolls = roll_d20(RollMode.NORMAL, ScriptedRng(13))
    assert (kept, rolls) == (13, (13,))


def test_advantage_rolls_two_and_keeps_the_higher():
    kept, rolls = roll_d20(RollMode.ADVANTAGE, ScriptedRng(4, 17))
    assert (kept, rolls) == (17, (4, 17))


def test_disadvantage_keeps_the_lower():
    kept, rolls = roll_d20(RollMode.DISADVANTAGE, ScriptedRng(4, 17))
    assert (kept, rolls) == (4, (4, 17))


def test_the_die_is_a_d20():
    rng = ScriptedRng(1)
    roll_d20(rng=rng)
    assert rng.asked == [(1, 20)]


def test_the_default_source_is_the_random_module():
    random.seed(7)
    expected = random.randint(1, 20)
    random.seed(7)
    assert roll_d20()[0] == expected


# --- ability checks ---------------------------------------------------------

def test_a_skill_check_uses_the_skill_modifier():
    result = ability_check(ROGUE, Ability.DEX, Skill.STEALTH, dc=15, rng=ScriptedRng(10))
    assert result.modifier == 3 + 4  # DEX +3, expertise 2 × 2
    assert result.total == 17
    assert result.success is True
    assert result.label == "Dexterity (Stealth)"


def test_the_skill_decides_the_ability_not_the_caller():
    result = ability_check(ROGUE, Ability.STR, Skill.STEALTH, rng=ScriptedRng(10))
    assert result.label == "Dexterity (Stealth)" and result.modifier == 7


def test_a_raw_ability_check_uses_the_ability_modifier():
    result = ability_check(ROGUE, Ability.STR, dc=10, rng=ScriptedRng(10))
    assert result.modifier == -1 and result.total == 9 and result.success is False


def test_meeting_the_dc_exactly_succeeds():
    assert ability_check(ROGUE, Ability.DEX, Skill.STEALTH, dc=17, rng=ScriptedRng(10)).success is True
    assert ability_check(ROGUE, Ability.DEX, Skill.STEALTH, dc=18, rng=ScriptedRng(10)).success is False


def test_no_dc_means_no_verdict():
    result = ability_check(ROGUE, Ability.DEX, Skill.STEALTH, rng=ScriptedRng(10))
    assert result.dc is None and result.success is None


def test_a_natural_20_is_not_an_automatic_success_on_a_check():
    result = ability_check(ROGUE, Ability.STR, dc=30, rng=ScriptedRng(20))
    assert result.natural_20 and result.success is False


def test_a_natural_1_is_not_an_automatic_failure_on_a_check():
    result = ability_check(ROGUE, Ability.DEX, Skill.STEALTH, dc=5, rng=ScriptedRng(1))
    assert result.natural_1 and result.success is True


def test_checks_honour_advantage():
    result = ability_check(ROGUE, Ability.DEX, Skill.STEALTH, mode=RollMode.ADVANTAGE, rng=ScriptedRng(3, 15))
    assert result.rolls == (3, 15) and result.kept == 15 and result.mode == RollMode.ADVANTAGE


def test_check_description_shows_the_whole_arithmetic():
    text = ability_check(ROGUE, Ability.DEX, Skill.STEALTH, dc=15, mode=RollMode.ADVANTAGE, rng=ScriptedRng(3, 10)).describe()
    assert text == "Dexterity (Stealth) check: d20 [3, 10]→10 + 7 = 17 (advantage) vs DC 15 — success"


def test_negative_modifiers_read_naturally():
    text = ability_check(ROGUE, Ability.STR, dc=10, rng=ScriptedRng(10)).describe()
    assert "d20 10 − 1 = 9 vs DC 10 — failure" in text


# --- saving throws ----------------------------------------------------------

def test_a_proficient_save_adds_the_bonus():
    result = saving_throw(ROGUE, Ability.DEX, dc=14, rng=ScriptedRng(9))
    assert result.modifier == 3 + 2 and result.total == 14 and result.success is True
    assert result.kind == "save" and result.label == "Dexterity"


def test_an_unproficient_save_does_not():
    result = saving_throw(ROGUE, Ability.WIS, dc=14, rng=ScriptedRng(9))
    assert result.modifier == 1 and result.success is False


# --- attacks ----------------------------------------------------------------

def test_a_hit_rolls_damage_with_the_ability_modifier_once():
    result = attack_roll(ROGUE, RAPIER, target_ac=14, rng=ScriptedRng(12, 6))
    assert result.roll.modifier == 3 + 2
    assert result.hit and not result.critical
    assert result.damage.rolls == (6,) and result.damage.modifier == 3 and result.damage.total == 9
    assert result.damage.damage_type == "piercing"


def test_a_miss_rolls_no_damage():
    rng = ScriptedRng(5)
    result = attack_roll(ROGUE, RAPIER, target_ac=14, rng=rng)
    assert result.roll.total == 10 and not result.hit and result.damage is None
    assert rng.values == []  # nothing more was rolled


def test_meeting_the_ac_hits():
    assert attack_roll(ROGUE, RAPIER, target_ac=14, rng=ScriptedRng(9, 1)).hit is True
    assert attack_roll(ROGUE, RAPIER, target_ac=15, rng=ScriptedRng(9)).hit is False


def test_a_natural_20_hits_and_crits_regardless_of_ac():
    result = attack_roll(ROGUE, RAPIER, target_ac=40, rng=ScriptedRng(20, 5, 7))
    assert result.hit and result.critical


def test_a_natural_1_misses_regardless_of_bonus():
    result = attack_roll(ROGUE, RAPIER, target_ac=1, rng=ScriptedRng(1))
    assert not result.hit and not result.critical and result.damage is None


def test_a_critical_doubles_the_dice_and_not_the_modifier():
    rng = ScriptedRng(20, 5, 7)
    result = attack_roll(ROGUE, RAPIER, target_ac=14, rng=rng)
    assert result.damage.dice == "2d8"
    assert result.damage.rolls == (5, 7)
    assert result.damage.modifier == 3
    assert result.damage.total == 5 + 7 + 3
    assert rng.asked[1:] == [(1, 8), (1, 8)]


def test_damage_never_goes_below_zero():
    weakling = ROGUE.model_copy(update={"abilities": Abilities(strength=3, dexterity=3)})
    damage = roll_damage(RAPIER, weakling.damage_bonus(RAPIER), rng=ScriptedRng(1))
    assert damage.modifier == -4 and damage.total == 0


def test_versatile_weapons_use_the_bigger_die_two_handed():
    fighter = ROGUE.model_copy(update={"abilities": Abilities(strength=16, dexterity=10)})
    rng = ScriptedRng(15, 9)
    result = attack_roll(fighter, LONGSWORD, target_ac=10, two_handed=True, rng=rng)
    assert result.damage.dice == "1d10" and rng.asked[1] == (1, 10)


def test_one_handed_uses_the_base_die():
    rng = ScriptedRng(15, 4)
    attack_roll(ROGUE, LONGSWORD, target_ac=10, rng=rng)
    assert rng.asked[1] == (1, 8)


def test_attack_description_is_complete():
    text = attack_roll(ROGUE, RAPIER, target_ac=14, rng=ScriptedRng(12, 6)).describe()
    assert text == "Rapier attack: d20 12 + 5 = 17 vs AC 14 — hit; damage 1d8 [6] + 3 = 9 piercing"


def test_a_critical_is_named_in_the_description():
    text = attack_roll(ROGUE, RAPIER, target_ac=14, rng=ScriptedRng(20, 5, 7)).describe()
    assert "critical hit" in text and "(critical: dice doubled)" in text


def test_results_are_immutable():
    result = ability_check(ROGUE, Ability.DEX, rng=ScriptedRng(10))
    with pytest.raises(Exception):
        result.total = 99
    assert isinstance(result, CheckResult)
    assert isinstance(attack_roll(ROGUE, RAPIER, 10, rng=ScriptedRng(10, 1)), AttackResult)


def test_a_seeded_random_replays_exactly():
    a = attack_roll(ROGUE, RAPIER, target_ac=12, rng=random.Random(42))
    b = attack_roll(ROGUE, RAPIER, target_ac=12, rng=random.Random(42))
    assert a == b
