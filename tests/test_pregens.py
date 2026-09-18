"""`src/engine/pregens.py` — six sheets a player can pick up cold."""

import pytest

pytest.importorskip("pydantic")

from src.engine.character import Ability, Character, Proficiency, Skill, ability_modifier  # noqa: E402
from src.engine.pregens import PREGENS, WEAPONS, describe_pregens, pregen  # noqa: E402


def test_there_are_six_and_they_are_keyed_by_class():
    assert set(PREGENS) == {"fighter", "rogue", "cleric", "wizard", "ranger", "barbarian"}
    for key, c in PREGENS.items():
        assert c.character_class.lower() == key


@pytest.mark.parametrize("key", list(PREGENS))
def test_every_pregen_is_level_one_and_round_trips(key):
    c = PREGENS[key]
    assert c.level == 1 and c.proficiency_bonus == 2
    assert Character.model_validate_json(c.model_dump_json()) == c


@pytest.mark.parametrize("key", list(PREGENS))
def test_level_one_hp_is_the_hit_die_plus_constitution(key):
    """Hill dwarves get +1 from Dwarven Toughness; everyone else is exact."""
    c = PREGENS[key]
    expected = c.hit_die + ability_modifier(c.abilities.constitution)
    assert c.max_hp in (expected, expected + 1)
    assert c.current_hp == c.max_hp


@pytest.mark.parametrize("key", list(PREGENS))
def test_every_pregen_has_two_save_proficiencies_and_at_least_two_skills(key):
    c = PREGENS[key]
    assert len(c.saves) == 2
    assert len(c.skills) >= 2


@pytest.mark.parametrize("key", list(PREGENS))
def test_every_pregen_carries_a_weapon_it_can_roll(key):
    c = PREGENS[key]
    assert c.weapons
    for weapon in c.weapons:
        # Proficiency is always added; the wizard's quarterstaff is +1 (STR 8).
        assert c.attack_bonus(weapon) == c.ability_modifier(c.attack_ability(weapon)) + 2


def test_the_rogue_sneaks_with_dexterity_and_double_proficiency():
    """The acceptance case from the spec: a Stealth check uses DEX + proficiency."""
    rogue = PREGENS["rogue"]
    assert rogue.skills[Skill.STEALTH] == Proficiency.EXPERTISE
    assert rogue.skill_modifier(Skill.STEALTH) == rogue.ability_modifier(Ability.DEX) + 2 * rogue.proficiency_bonus
    assert rogue.skill_modifier(Skill.STEALTH) == 7


def test_the_rogue_stabs_with_dexterity():
    rogue = PREGENS["rogue"]
    assert rogue.attack_ability(rogue.weapon("rapier")) == Ability.DEX
    assert rogue.attack_bonus(rogue.weapon("rapier")) == 5


def test_the_barbarian_swings_with_strength():
    barbarian = PREGENS["barbarian"]
    assert barbarian.attack_bonus(barbarian.weapon("greataxe")) == 3 + 2
    assert barbarian.armor_class == 10 + 1 + 3  # Unarmored Defense


def test_the_fighter_is_the_tank():
    fighter = PREGENS["fighter"]
    assert fighter.armor_class == max(c.armor_class for c in PREGENS.values())


def test_the_wizard_is_the_squishiest():
    wizard = PREGENS["wizard"]
    assert wizard.max_hp == min(c.max_hp for c in PREGENS.values())


def test_pregen_hands_out_a_copy_with_the_players_identity():
    mine = pregen("rogue", player_id="discord:123", name="Pip")
    assert (mine.player_id, mine.name) == ("discord:123", "Pip")
    assert PREGENS["rogue"].player_id == "template" and PREGENS["rogue"].name == "Kara Swiftfoot"
    assert mine.weapons == PREGENS["rogue"].weapons


def test_pregen_keeps_the_template_name_when_none_is_given():
    assert pregen("Fighter ", player_id="x").name == "Dorn Ironfist"


def test_an_unknown_pregen_lists_the_choices():
    with pytest.raises(KeyError) as info:
        pregen("paladin", player_id="x")
    assert "fighter" in str(info.value) and "barbarian" in str(info.value)


def test_weapon_table_matches_the_srd():
    assert WEAPONS["longsword"].versatile_dice == "1d10"
    assert WEAPONS["greataxe"].damage_dice == "1d12" and WEAPONS["greataxe"].heavy
    assert WEAPONS["rapier"].finesse and not WEAPONS["rapier"].ranged
    assert WEAPONS["longbow"].ranged and WEAPONS["longbow"].damage_dice == "1d8"


def test_describe_pregens_names_each_one():
    lines = describe_pregens()
    assert len(lines) == 6
    assert any(line.startswith("rogue: Kara Swiftfoot") and "AC 14" in line for line in lines)
