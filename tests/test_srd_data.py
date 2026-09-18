"""`src/srd/data.py` — the vendored SRD as data, looked up by name."""

import pytest

from src.srd import UnknownEntry, condition, equipment, find, magic_item, monster, names, spell
from src.srd.data import levenshtein


# --- exact --------------------------------------------------------------------

def test_the_goblin_is_where_the_spec_says():
    g = monster("goblin")
    assert g["name"] == "Goblin"
    assert g["hit_points"] == 7
    assert g["armor_class"][0]["value"] == 15


def test_lookup_is_case_insensitive():
    assert monster("GOBLIN") is monster("Goblin") is monster("goblin")


def test_lookup_accepts_the_slug_or_the_display_name():
    assert monster("adult-red-dragon")["name"] == "Adult Red Dragon"
    assert monster("Adult Red Dragon")["index"] == "adult-red-dragon"


def test_entries_are_loaded_once():
    assert monster("ogre") is monster("ogre")


def test_the_bestiary_is_the_whole_file():
    assert len(names("Monsters")) == 334
    assert "Goblin" in names("Monsters")


# --- fuzzy --------------------------------------------------------------------

@pytest.mark.parametrize("typo", ["gobln", "goblim", "gobliin", "Goblinn"])
def test_a_misspelling_within_two_edits_still_resolves(typo):
    assert monster(typo)["name"] == "Goblin"


def test_ties_go_to_the_more_similar_name():
    assert monster("ogr")["name"] == "Ogre"  # not the orc, also one edit away
    assert monster("orcc")["name"] == "Orc"


def test_three_edits_is_a_different_creature():
    with pytest.raises(UnknownEntry):
        monster("gobbbln")


def test_an_unknown_name_raises_with_suggestions():
    with pytest.raises(UnknownEntry) as info:
        monster("beholder")  # product identity, not in the SRD
    assert "beholder" in str(info.value)
    assert "Did you mean" in str(info.value)


def test_a_blank_name_is_an_error_not_a_random_monster():
    with pytest.raises(UnknownEntry):
        monster("   ")


@pytest.mark.parametrize("a,b,d", [
    ("goblin", "goblin", 0), ("gobln", "goblin", 1), ("gobliim", "goblin", 2),
    ("ogre", "orc", 2), ("a", "abcd", 3), ("kitten", "sitting", 3),
])
def test_levenshtein(a, b, d):
    assert levenshtein(a, b, limit=10) == d


def test_levenshtein_gives_up_past_the_limit():
    assert levenshtein("goblin", "adult red dragon") == 3  # limit + 1, not the true 13


# --- the other tables -----------------------------------------------------------

def test_spells():
    f = spell("fireball")
    assert f["level"] == 3
    assert f["damage"]["damage_at_slot_level"]["3"] == "8d6"
    assert spell("fire ball")["name"] == "Fireball"


def test_equipment():
    assert equipment("longsword")["damage"]["damage_dice"] == "1d8"
    assert equipment("Longsword")["two_handed_damage"]["damage_dice"] == "1d10"


def test_conditions_carry_their_rules_text():
    prone = condition("prone")
    assert any("disadvantage on attack rolls" in line for line in prone["desc"])


def test_magic_items():
    assert magic_item("bag of holding")["name"] == "Bag of Holding"


def test_find_names_the_category_in_its_error():
    with pytest.raises(UnknownEntry) as info:
        find("Spells", "wingardium leviosa")
    assert "No spell called" in str(info.value)


def test_a_missing_file_says_where_the_corpus_lives():
    with pytest.raises(FileNotFoundError) as info:
        find("Feats", "grappler")
    assert "corpus/srd" in str(info.value)
