"""Tests for the SRD corpus loader.

These read the real vendored corpus in `corpus/srd`, which ships with the
repository — no network, no embedding model, no daemon.
"""

import pytest

from src.agents.researcher import ResearcherAgent
from src.config import SRD_DIRECTORY
from src.data.srd_loader import (
    SRD_FILES,
    chunk_entry,
    load_srd_documents,
    merge_shared_entries,
    render_monster,
    render_spell,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def documents():
    return load_srd_documents(SRD_DIRECTORY)


def find(documents, name):
    return [d for d in documents if d.metadata.get("name") == name]


# --- the corpus loads -------------------------------------------------------

def test_the_vendored_corpus_loads(documents):
    assert len(documents) > 3000


def test_every_category_is_present(documents):
    categories = {d.metadata["category"] for d in documents}
    for expected in ["Monsters", "Spells", "Rules", "Class Features", "Conditions"]:
        assert expected in categories


def test_every_chunk_is_attributable(documents):
    """A chunk with no name cannot be cited, which is the point of this corpus."""
    for doc in documents:
        assert doc.metadata["source"] == "SRD 5.1"
        assert doc.metadata["name"]
        assert doc.metadata["category"]
        assert doc.page_content.strip()


def test_missing_corpus_says_where_to_look():
    with pytest.raises(FileNotFoundError, match="corpus/srd"):
        load_srd_documents("nowhere")


# --- rendering --------------------------------------------------------------

def test_a_monster_renders_as_a_stat_block(documents):
    goblin = find(documents, "Goblin")[0].page_content

    assert goblin.startswith("# Goblin")
    assert "Armor Class 15" in goblin
    assert "Hit Points 7" in goblin
    assert "Challenge 0.25" in goblin
    assert "Nimble Escape" in goblin
    # The OCR'd rulebooks rendered this as "ld6" — an l for a 1, in the one
    # token a dice-rolling assistant cannot afford to get wrong.
    assert "1d6 + 2" in goblin
    assert "ld6" not in goblin


def test_a_monster_fits_in_one_chunk(documents):
    """Splitting a stat block puts a creature's actions in a different chunk
    from its hit points."""
    assert len(find(documents, "Goblin")) == 1


def test_a_spell_carries_its_casting_details(documents):
    fireball = find(documents, "Fireball")[0]

    assert "Level 3 evocation" in fireball.page_content
    assert "Casting Time: 1 action" in fireball.page_content
    assert "8d6" in fireball.page_content
    assert fireball.metadata["level"] == "3"
    assert fireball.metadata["school"] == "Evocation"


def test_monster_metadata_supports_filtering(documents):
    goblin = find(documents, "Goblin")[0]
    assert goblin.metadata["cr"] == "0.25"
    assert goblin.metadata["type"] == "humanoid"


def test_descriptions_keep_their_punctuation(documents):
    """An early version stripped trailing periods off every action."""
    goblin = find(documents, "Goblin")[0].page_content
    assert "slashing damage." in goblin


def test_renderers_tolerate_an_empty_entry():
    assert render_monster({}) is not None
    assert render_spell({"level": 0, "school": {"name": "Evocation"}}) is not None


# --- chunking ---------------------------------------------------------------

def test_a_long_entry_is_split_and_every_piece_keeps_its_name():
    """Chroma embeds page_content and never metadata, so a piece without its
    title cannot be found by name."""
    pieces = chunk_entry("word " * 900, "Making an Attack", 1000, 200)

    assert len(pieces) > 1
    assert all(p.startswith("# Making an Attack") for p in pieces)


def test_a_short_entry_is_not_split():
    assert chunk_entry("short", "Goblin", 1000, 200) == ["# Goblin\nshort"]


def test_rule_sections_are_split_but_stay_findable(documents):
    attack = find(documents, "Making an Attack")
    assert len(attack) > 1
    assert all(d.page_content.startswith("# Making an Attack") for d in attack)


# --- deduplication ----------------------------------------------------------

def test_shared_features_are_merged_not_repeated():
    """`Ability Score Improvement` is stored once per class per level — 50
    identical copies, which crowded the top of unrelated result sets."""
    entries = [
        {"name": "Ability Score Improvement", "desc": ["Increase a score."],
         "class": {"name": "Rogue"}, "level": 4},
        {"name": "Ability Score Improvement", "desc": ["Increase a score."],
         "class": {"name": "Bard"}, "level": 4},
        {"name": "Sneak Attack", "desc": ["Extra damage."],
         "class": {"name": "Rogue"}, "level": 1},
    ]
    merged = merge_shared_entries(entries, "Features")

    assert len(merged) == 2
    assert merged[0]["_owners"] == ["Rogue", "Bard"]


def test_differing_text_is_not_merged():
    entries = [
        {"name": "Extra Attack", "desc": ["Twice."], "class": {"name": "Fighter"}},
        {"name": "Extra Attack", "desc": ["Three times."], "class": {"name": "Monk"}},
    ]
    assert len(merge_shared_entries(entries, "Features")) == 2


def test_merging_only_applies_where_sharing_happens():
    entries = [{"name": "Goblin"}, {"name": "Goblin"}]
    assert len(merge_shared_entries(entries, "Monsters")) == 2


def test_the_merged_feature_names_its_classes(documents):
    improvement = find(documents, "Ability Score Improvement")
    assert len(improvement) < 10, "50 near-identical copies crowd retrieval"
    assert any("Classes:" in d.page_content for d in improvement)


# --- citations --------------------------------------------------------------

def test_an_srd_chunk_cites_by_name(documents):
    goblin = find(documents, "Goblin")[0]
    assert ResearcherAgent.citation_for(goblin) == "SRD 5.1, Monsters: Goblin"


def test_pdf_chunks_still_cite_by_page():
    """The rulebook index is still buildable, so both shapes must work."""
    from langchain_core.documents import Document

    doc = Document(page_content="x",
                   metadata={"book": "Player's Handbook", "page_number": 89})
    assert ResearcherAgent.citation_for(doc) == "Player's Handbook, p.89"


def test_configured_files_all_exist():
    from pathlib import Path

    for stem in SRD_FILES:
        assert (Path(SRD_DIRECTORY) / f"{stem}.json").is_file(), stem
