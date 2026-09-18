"""`src/engine/combatant.py` and `src/srd/bestiary.py` — stat blocks as combatants."""

import random

import pytest

from src.engine.character import Ability, Condition
from src.engine.combatant import Attack, Combatant, Damage
from src.srd import monster
from src.srd.bestiary import summon, summon_group


# --- the acceptance case ------------------------------------------------------

def test_the_goblin_from_the_spec():
    g = summon("goblin")
    assert g.armor_class == 15
    assert g.max_hp == 7 and g.current_hp == 7
    scimitar = g.attack("scimitar")
    assert scimitar.attack_bonus == 4
    assert [d.notation for d in scimitar.damage] == ["1d6+2"]
    assert scimitar.damage[0].damage_type == "slashing"
    assert not scimitar.ranged
    assert g.attack("Shortbow").ranged


def test_the_goblin_carries_its_stat_block():
    g = summon("goblin")
    assert g.id == "goblin" and g.name == "Goblin" and g.kind == "monster"
    assert (g.size, g.creature_type, g.alignment) == ("Small", "humanoid", "neutral evil")
    assert g.challenge_rating == 0.25 and g.xp == 50
    assert g.speed == {"walk": "30 ft."}
    assert g.senses["darkvision"] == "60 ft."
    assert any(t.startswith("Nimble Escape.") for t in g.traits)
    assert g.multiattack is None


def test_listed_skills_and_saves_are_totals_and_the_rest_are_modifiers():
    g = summon("goblin")
    assert g.skill_bonuses == {"Stealth": 6}
    assert g.skill_modifier("Stealth") == 6
    assert g.skill_modifier("Perception") == -1  # WIS 8, unlisted
    assert g.save_modifier(Ability.DEX) == 2  # no listed save → DEX 14 → +2
    assert g.initiative_modifier == 2


def test_a_dragon_has_listed_saves_and_legendary_actions():
    d = summon("adult red dragon")
    assert d.save_bonuses[Ability.DEX] > d.ability_modifier(Ability.DEX)
    assert d.legendary_actions
    assert d.damage_immunities == ["fire"]


# --- damage shapes ------------------------------------------------------------

@pytest.mark.parametrize("notation,dice,bonus,back", [
    ("1d6+2", "1d6", 2, "1d6+2"),
    ("2d8", "2d8", 0, "2d8"),
    ("1", None, 1, "1"),
    ("3d6 + 4", "3d6", 4, "3d6+4"),
    ("1d4-1", "1d4", -1, "1d4-1"),
])
def test_damage_notation_round_trips(notation, dice, bonus, back):
    d = Damage.parse(notation, "piercing")
    assert (d.dice, d.bonus, d.notation) == (dice, bonus, back)


def test_unreadable_damage_is_rejected():
    with pytest.raises(ValueError):
        Damage.parse("lots", "fire")


def test_average_damage():
    assert Damage.parse("1d6+2", "slashing").average == 5
    assert Damage.parse("2d8+5", "slashing").average == 14
    assert Damage.parse("1", "piercing").average == 1


def test_a_rats_bite_is_flat_damage():
    bite = summon("rat").attack("bite")
    assert bite.damage[0].dice is None and bite.damage[0].bonus == 1


def test_a_dragons_bite_has_two_damage_components():
    bite = summon("adult black dragon").attack("bite")
    assert [d.damage_type for d in bite.damage] == ["piercing", "acid"]
    assert [d.notation for d in bite.damage] == ["2d10+6", "1d8"]


def test_a_giant_spiders_poison_is_a_rider_kept_in_the_description():
    """The poison is a saving throw, not a damage component, so it lives in `desc`."""
    bite = summon("giant spider").attack("bite")
    assert [d.notation for d in bite.damage] == ["1d8+3"]
    assert "DC 11 Constitution saving throw" in bite.desc and "2d8" in bite.desc


def test_a_choice_of_damage_takes_the_first_option():
    scimitar = summon("djinni").attack("scimitar")
    assert scimitar.damage and scimitar.damage[0].damage_type in {"slashing", "lightning", "thunder"}


def test_multiattack_is_kept_as_text_and_not_as_an_attack():
    owlbear = summon("owlbear")
    assert owlbear.multiattack.startswith("The owlbear makes two attacks")
    assert [a.name for a in owlbear.attacks] == ["Beak", "Claws"]


def test_attack_lookup_names_what_is_available():
    with pytest.raises(KeyError) as info:
        summon("goblin").attack("breath weapon")
    assert "Scimitar" in str(info.value)


def test_attack_describe():
    assert summon("goblin").attack("scimitar").describe() == "Scimitar (melee, +4 to hit, 1d6+2 slashing)"


# --- hit points ----------------------------------------------------------------

def test_hp_is_the_listed_average_by_default():
    assert summon("ogre").max_hp == 59


def test_rolled_hp_adds_constitution_per_die_and_replays_under_a_seed():
    a = summon("ogre", rng=random.Random(1), roll_hp=True)
    b = summon("ogre", rng=random.Random(1), roll_hp=True)
    assert a.max_hp == b.max_hp
    assert 7 + 21 <= a.max_hp <= 70 + 21  # 7d10 + 7 × CON(+3)


def test_rolled_hp_never_drops_below_one():
    entry = dict(monster("rat"))
    entry["constitution"] = 1  # −5 per die would go negative
    assert Combatant.from_monster(entry, roll_hp=True, rng=random.Random(0)).max_hp >= 1


# --- identity and groups --------------------------------------------------------

def test_an_id_can_be_given():
    assert summon("goblin", combatant_id="boss").id == "boss"


def test_a_group_is_numbered():
    pack = summon_group("goblin", 3)
    assert [g.id for g in pack] == ["goblin-1", "goblin-2", "goblin-3"]
    assert all(g.name == "Goblin" for g in pack)


def test_sheet_line():
    g = summon("goblin", combatant_id="goblin-2").model_copy(update={"current_hp": 3, "conditions": [Condition.PRONE]})
    assert g.sheet_line() == "goblin-2 (Goblin): AC 15, 3/7 HP, prone"


# --- contract -------------------------------------------------------------------

def test_json_round_trip_is_lossless():
    for name in ["goblin", "adult red dragon", "giant spider", "djinni"]:
        c = summon(name)
        assert Combatant.model_validate_json(c.model_dump_json()) == c


def test_combatants_are_frozen_and_bounded():
    g = summon("goblin")
    with pytest.raises(Exception):
        g.current_hp = 1
    with pytest.raises(ValueError):
        g.model_copy(update={"current_hp": 99}).model_validate(g.model_copy(update={"current_hp": 99}).model_dump())


def test_every_srd_monster_builds():
    """No stat block shape in the corpus escapes the loader."""
    from src.srd import names

    built = [summon(n) for n in names("Monsters")]
    assert len(built) == 334
    assert all(c.max_hp >= 1 and c.armor_class >= 5 for c in built)
    assert sum(1 for c in built if c.attacks) > 300


def test_attack_model_is_frozen():
    a = Attack(name="Bite", attack_bonus=3)
    with pytest.raises(Exception):
        a.attack_bonus = 9
