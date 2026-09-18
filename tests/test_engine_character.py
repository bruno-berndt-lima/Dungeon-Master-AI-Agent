"""`src/engine/character.py` — the sheet and every number derived from it.

Pure Python plus pydantic; no model, no graph.
"""

import pytest

pytest.importorskip("pydantic")

from src.engine.character import (  # noqa: E402
    SKILL_ABILITY,
    Abilities,
    Ability,
    Character,
    Condition,
    Proficiency,
    Skill,
    Weapon,
    ability_modifier,
    proficiency_bonus_for,
)


def make(**overrides) -> Character:
    base = dict(
        player_id="p1", name="Test", race="Human", character_class="Fighter", level=1,
        abilities=Abilities(strength=16, dexterity=14, constitution=13, intelligence=10, wisdom=12, charisma=8),
        armor_class=16, max_hp=12, hit_die=10,
    )
    base.update(overrides)
    return Character(**base)


LONGSWORD = Weapon(name="Longsword", damage_dice="1d8", damage_type="slashing", versatile_dice="1d10")
RAPIER = Weapon(name="Rapier", damage_dice="1d8", damage_type="piercing", finesse=True)
SHORTBOW = Weapon(name="Shortbow", damage_dice="1d6", damage_type="piercing", ranged=True)
JAVELIN = Weapon(name="Javelin", damage_dice="1d6", damage_type="piercing", thrown=True)


# --- the two tables everything else rests on --------------------------------

@pytest.mark.parametrize("score,expected", [
    (1, -5), (3, -4), (8, -1), (9, -1), (10, 0), (11, 0), (12, 1), (13, 1),
    (15, 2), (17, 3), (18, 4), (20, 5), (30, 10),
])
def test_ability_modifier_table(score, expected):
    assert ability_modifier(score) == expected


@pytest.mark.parametrize("level,expected", [
    (1, 2), (4, 2), (5, 3), (8, 3), (9, 4), (12, 4), (13, 5), (16, 5), (17, 6), (20, 6),
])
def test_proficiency_bonus_by_level(level, expected):
    assert proficiency_bonus_for(level) == expected


def test_every_skill_has_an_ability():
    assert set(SKILL_ABILITY) == set(Skill)


# --- construction -----------------------------------------------------------

def test_a_fresh_sheet_is_at_full_health_with_all_hit_dice():
    c = make(level=3, max_hp=28)
    assert c.current_hp == 28
    assert c.hit_dice_remaining == 3
    assert c.temp_hp == 0


def test_current_hp_above_max_is_rejected():
    with pytest.raises(ValueError):
        make(max_hp=10, current_hp=11)


def test_negative_hp_is_rejected():
    with pytest.raises(ValueError):
        make(current_hp=-1)


def test_ability_scores_are_bounded():
    with pytest.raises(ValueError):
        Abilities(strength=0)
    with pytest.raises(ValueError):
        Abilities(charisma=31)


def test_weapon_dice_must_be_rollable():
    with pytest.raises(ValueError):
        Weapon(name="Broken", damage_dice="lots", damage_type="slashing")


def test_the_sheet_is_frozen():
    c = make()
    with pytest.raises(Exception):
        c.current_hp = 5


def test_changes_go_through_model_copy():
    c = make()
    hurt = c.model_copy(update={"current_hp": 4})
    assert (c.current_hp, hurt.current_hp) == (12, 4)


# --- JSON round trip: it lives inside the checkpoint ------------------------

def test_json_round_trip_is_lossless():
    c = make(
        skills={Skill.STEALTH: Proficiency.EXPERTISE, Skill.ATHLETICS: Proficiency.PROFICIENT},
        saves=[Ability.STR, Ability.CON],
        conditions=[Condition.PRONE],
        weapons=[LONGSWORD, RAPIER],
        inventory=["torch", "rope"],
        temp_hp=3,
    )
    assert Character.model_validate_json(c.model_dump_json()) == c


def test_enum_keys_survive_a_plain_dict_round_trip():
    c = make(skills={Skill.SLEIGHT_OF_HAND: Proficiency.PROFICIENT})
    again = Character.model_validate(c.model_dump(mode="json"))
    assert again.skills == {Skill.SLEIGHT_OF_HAND: Proficiency.PROFICIENT}


# --- derived modifiers ------------------------------------------------------

def test_ability_modifiers_read_the_scores():
    c = make()
    assert c.ability_modifier(Ability.STR) == 3
    assert c.ability_modifier(Ability.CHA) == -1


def test_skill_modifier_without_proficiency_is_the_ability_modifier():
    assert make().skill_modifier(Skill.ATHLETICS) == 3


def test_proficiency_adds_the_bonus_once():
    c = make(skills={Skill.ATHLETICS: Proficiency.PROFICIENT})
    assert c.skill_modifier(Skill.ATHLETICS) == 3 + 2


def test_expertise_adds_the_bonus_twice():
    c = make(skills={Skill.STEALTH: Proficiency.EXPERTISE})
    assert c.skill_modifier(Skill.STEALTH) == 2 + 4  # DEX 14 → +2, proficiency 2 × 2


def test_proficiency_bonus_scales_with_level():
    c = make(level=5, skills={Skill.ATHLETICS: Proficiency.PROFICIENT})
    assert c.proficiency_bonus == 3
    assert c.skill_modifier(Skill.ATHLETICS) == 3 + 3


def test_save_modifier_adds_proficiency_only_for_listed_saves():
    c = make(saves=[Ability.STR])
    assert c.save_modifier(Ability.STR) == 3 + 2
    assert c.save_modifier(Ability.DEX) == 2


def test_passive_perception_is_ten_plus_the_modifier():
    c = make(skills={Skill.PERCEPTION: Proficiency.PROFICIENT})
    assert c.passive_perception == 10 + 1 + 2  # WIS 12 → +1


def test_initiative_is_dexterity():
    assert make().initiative_modifier == 2


def test_string_inputs_are_accepted_for_enums():
    c = make(skills={"Stealth": "expertise"}, saves=["DEX"])
    assert c.skill_modifier("Stealth") == 6
    assert c.save_modifier("DEX") == 4


# --- weapons ----------------------------------------------------------------

def test_melee_weapons_use_strength():
    c = make()
    assert c.attack_ability(LONGSWORD) == Ability.STR
    assert c.attack_bonus(LONGSWORD) == 3 + 2
    assert c.damage_bonus(LONGSWORD) == 3


def test_ranged_weapons_use_dexterity():
    c = make()
    assert c.attack_ability(SHORTBOW) == Ability.DEX
    assert c.attack_bonus(SHORTBOW) == 2 + 2


def test_finesse_picks_the_better_of_str_and_dex():
    strong = make()
    assert strong.attack_ability(RAPIER) == Ability.STR
    nimble = make(abilities=Abilities(strength=8, dexterity=17))
    assert nimble.attack_ability(RAPIER) == Ability.DEX
    assert nimble.attack_bonus(RAPIER) == 3 + 2


def test_thrown_without_finesse_is_still_strength():
    assert make().attack_ability(JAVELIN) == Ability.STR


def test_weapon_lookup_is_case_insensitive_and_names_what_is_carried():
    c = make(weapons=[LONGSWORD])
    assert c.weapon("longsword") is LONGSWORD
    with pytest.raises(KeyError) as info:
        c.weapon("greataxe")
    assert "Longsword" in str(info.value)


# --- state ------------------------------------------------------------------

def test_conscious_needs_hp_and_no_unconscious_condition():
    assert make().is_conscious
    assert not make(current_hp=0).is_conscious
    assert not make(conditions=[Condition.UNCONSCIOUS]).is_conscious


def test_sheet_line_carries_what_the_dm_needs():
    line = make(current_hp=5, temp_hp=2, conditions=[Condition.POISONED]).sheet_line()
    assert "AC 16" in line and "5/12 HP" in line and "+2 temp" in line and "poisoned" in line


def test_the_engine_does_not_import_the_agents_or_langchain():
    import pathlib
    import re

    forbidden = re.compile(r"^\s*(from|import)\s+(langchain|langgraph|src\.agents|src\.graph)", re.MULTILINE)
    for module in pathlib.Path("src/engine").glob("*.py"):
        assert not forbidden.search(module.read_text()), module
