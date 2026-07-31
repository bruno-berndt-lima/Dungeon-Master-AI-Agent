"""Turn the vendored SRD 5.1 JSON into retrievable documents.

The PDF path (`loader.py`) produces one document per *page*, which is an artifact
of how a book is printed rather than how rules are asked about: a chunk boundary
lands mid-stat-block, and the only metadata available is a page index. This
loader produces one document per *entry* — a monster, a spell, a rule section —
so a chunk is a thing you can name.

Two consequences worth knowing:

- **Citations become names.** `SRD 5.1, Monsters: Goblin` instead of `p.261`. A
  reader can check that; a PDF page index is offset from the printed page number
  and cannot be checked (KNOWN_ISSUES #28).
- **Every chunk repeats its entry's name.** Chroma embeds `page_content` only,
  never metadata, so a long entry split across chunks would leave the later
  chunks unsearchable by name. `chunk_entry` re-heads each piece.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SRD_SOURCE = "SRD 5.1"

# Files worth indexing, and the category name used in citations. Left out on
# purpose: Equipment-Categories and Proficiencies, which are mostly cross
# reference lists whose prose adds nothing a player would ask about.
SRD_FILES = {
    "Monsters": "Monsters",
    "Spells": "Spells",
    "Magic-Items": "Magic Items",
    "Equipment": "Equipment",
    "Classes": "Classes",
    "Subclasses": "Subclasses",
    "Features": "Class Features",
    "Levels": "Class Levels",
    "Races": "Races",
    "Traits": "Racial Traits",
    "Backgrounds": "Backgrounds",
    "Rule-Sections": "Rules",
    "Rules": "Rules",
    "Conditions": "Conditions",
    "Skills": "Skills",
    "Ability-Scores": "Ability Scores",
    "Damage-Types": "Damage Types",
    "Weapon-Properties": "Weapon Properties",
    "Magic-Schools": "Magic Schools",
    "Alignments": "Alignments",
    "Languages": "Languages",
}

ABILITIES = [
    ("strength", "STR"), ("dexterity", "DEX"), ("constitution", "CON"),
    ("intelligence", "INT"), ("wisdom", "WIS"), ("charisma", "CHA"),
]


def _name_of(value: Any) -> str:
    """SRD references are `{"index":..., "name":..., "url":...}`; take the name."""
    if isinstance(value, dict):
        return str(value.get("name", "")).strip()
    return str(value).strip()


def _names(values: Optional[Iterable]) -> str:
    return ", ".join(n for n in (_name_of(v) for v in values or []) if n)


def _modifier(score: Optional[int]) -> str:
    if score is None:
        return ""
    mod = (int(score) - 10) // 2
    return f"{score} ({mod:+d})"


def _armor_class(value: Any) -> str:
    """AC is a plain int in older dumps and a list of typed entries in newer ones."""
    if isinstance(value, list):
        parts = []
        for entry in value:
            if not isinstance(entry, dict):
                continue
            note = _names(entry.get("armor")) or entry.get("condition", {}).get("name", "")
            parts.append(f"{entry.get('value')}" + (f" ({note})" if note else ""))
        return ", ".join(parts)
    return str(value or "")


def _block(title: str, entries: Optional[Iterable[Dict]]) -> str:
    """A named list of abilities/actions, the shape most SRD sub-lists share."""
    lines = []
    for entry in entries or []:
        name = str(entry.get("name", "")).strip()
        desc = str(entry.get("desc", "")).strip()
        if not (name or desc):
            continue
        lines.append(f"{name}. {desc}" if name and desc else (name or desc))
    if not lines:
        return ""
    return f"{title}\n" + "\n".join(lines) if title else "\n".join(lines)


def render_monster(entry: Dict) -> str:
    speed = ", ".join(f"{k} {v}" for k, v in (entry.get("speed") or {}).items())
    scores = " ".join(
        f"{label} {_modifier(entry.get(key))}" for key, label in ABILITIES
    )

    subtype = f" ({entry['subtype']})" if entry.get("subtype") else ""
    lines = [
        f"{entry.get('size', '')} {entry.get('type', '')}{subtype}, "
        f"{entry.get('alignment', '')}".strip(),
        f"Armor Class {_armor_class(entry.get('armor_class'))}",
        f"Hit Points {entry.get('hit_points')} ({entry.get('hit_dice')})",
        f"Speed {speed}" if speed else "",
        scores,
    ]

    for label, key in [
        ("Damage Vulnerabilities", "damage_vulnerabilities"),
        ("Damage Resistances", "damage_resistances"),
        ("Damage Immunities", "damage_immunities"),
    ]:
        value = entry.get(key)
        text = ", ".join(value) if isinstance(value, list) else str(value or "")
        if text:
            lines.append(f"{label} {text}")

    conditions = _names(entry.get("condition_immunities"))
    if conditions:
        lines.append(f"Condition Immunities {conditions}")

    senses = entry.get("senses") or {}
    if senses:
        lines.append("Senses " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in senses.items()))
    if entry.get("languages"):
        lines.append(f"Languages {entry['languages']}")
    lines.append(
        f"Challenge {entry.get('challenge_rating')} ({entry.get('xp', 0)} XP)"
    )

    for title, key in [
        ("", "special_abilities"),
        ("Actions", "actions"),
        ("Reactions", "reactions"),
        ("Legendary Actions", "legendary_actions"),
    ]:
        block = _block(title, entry.get(key))
        if block:
            lines.append("")
            lines.append(block)

    return "\n".join(line for line in lines if line != "" or True).strip()


def render_spell(entry: Dict) -> str:
    level = entry.get("level")
    school = _name_of(entry.get("school"))
    heading = (
        f"{school} cantrip" if level == 0 else f"Level {level} {school.lower()}"
    )
    lines = [
        heading,
        f"Casting Time: {entry.get('casting_time')}",
        f"Range: {entry.get('range')}",
        f"Components: {', '.join(entry.get('components') or [])}"
        + (f" ({entry['material']})" if entry.get("material") else ""),
        f"Duration: {entry.get('duration')}"
        + (" (concentration)" if entry.get("concentration") else ""),
    ]
    if entry.get("ritual"):
        lines.append("Ritual: yes")
    classes = _names(entry.get("classes"))
    if classes:
        lines.append(f"Classes: {classes}")

    lines.append("")
    lines.extend(entry.get("desc") or [])
    if entry.get("higher_level"):
        lines.append("")
        lines.append("At Higher Levels. " + " ".join(entry["higher_level"]))
    return "\n".join(str(line) for line in lines).strip()


def render_equipment(entry: Dict) -> str:
    lines = [_name_of(entry.get("equipment_category"))]
    cost = entry.get("cost") or {}
    if cost:
        lines.append(f"Cost: {cost.get('quantity')} {cost.get('unit')}")
    damage = entry.get("damage") or {}
    if damage:
        lines.append(
            f"Damage: {damage.get('damage_dice')} "
            f"{_name_of(damage.get('damage_type'))}"
        )
    if entry.get("weight"):
        lines.append(f"Weight: {entry['weight']} lb.")
    properties = _names(entry.get("properties"))
    if properties:
        lines.append(f"Properties: {properties}")
    if entry.get("desc"):
        lines.append("")
        lines.extend(entry["desc"])
    return "\n".join(str(line) for line in lines).strip()


def render_magic_item(entry: Dict) -> str:
    rarity = (entry.get("rarity") or {}).get("name", "")
    lines = [
        f"{_name_of(entry.get('equipment_category'))}, {rarity}".strip(", "),
        "",
    ]
    lines.extend(entry.get("desc") or [])
    return "\n".join(str(line) for line in lines).strip()


def render_class(entry: Dict) -> str:
    lines = [
        f"Hit Die: d{entry.get('hit_die')}",
        f"Saving Throw Proficiencies: {_names(entry.get('saving_throws'))}",
        f"Proficiencies: {_names(entry.get('proficiencies'))}",
    ]
    subclasses = _names(entry.get("subclasses"))
    if subclasses:
        lines.append(f"Subclasses: {subclasses}")
    multi = (entry.get("multi_classing") or {}).get("prerequisites") or []
    if multi:
        lines.append(
            "Multiclassing prerequisites: "
            + ", ".join(
                f"{_name_of(p.get('ability_score'))} {p.get('minimum_score')}"
                for p in multi
            )
        )
    return "\n".join(lines).strip()


def render_race(entry: Dict) -> str:
    bonuses = ", ".join(
        f"{_name_of(b.get('ability_score'))} +{b.get('bonus')}"
        for b in entry.get("ability_bonuses") or []
    )
    lines = [
        f"Size: {entry.get('size')}. {entry.get('size_description', '')}",
        f"Speed: {entry.get('speed')}",
        f"Ability Score Increase: {bonuses}" if bonuses else "",
        f"Age: {entry.get('age', '')}",
        f"Alignment: {entry.get('alignment', '')}",
        f"Languages: {entry.get('language_desc', '')}",
        f"Traits: {_names(entry.get('traits'))}",
    ]
    subraces = _names(entry.get("subraces"))
    if subraces:
        lines.append(f"Subraces: {subraces}")
    return "\n".join(line for line in lines if line.strip()).strip()


def render_level(entry: Dict) -> str:
    features = _names(entry.get("features"))
    lines = [
        f"Proficiency Bonus: +{entry.get('prof_bonus')}",
        f"Features gained: {features}" if features else "",
    ]
    specific = entry.get("class_specific") or {}
    for key, value in specific.items():
        if isinstance(value, (int, str)):
            lines.append(f"{key.replace('_', ' ').title()}: {value}")
    spellcasting = entry.get("spellcasting") or {}
    if spellcasting:
        slots = ", ".join(
            f"{k.replace('_', ' ')} {v}" for k, v in spellcasting.items() if v
        )
        lines.append(f"Spellcasting: {slots}")
    return "\n".join(line for line in lines if line.strip()).strip()


def render_prose(entry: Dict) -> str:
    """Rule sections, conditions, skills, traits, features — a `desc` and little else."""
    desc = entry.get("desc")
    if isinstance(desc, list):
        body = "\n".join(str(line) for line in desc)
    else:
        body = str(desc or "")

    prefix = []
    owners = entry.get("_owners")
    if owners:
        label = "Classes" if len(owners) > 1 else "Class"
        prefix.append(f"{label}: {', '.join(owners)}")
    elif entry.get("class"):
        prefix.append(f"Class: {_name_of(entry['class'])}")
    if entry.get("level"):
        prefix.append(f"Level: {entry['level']}")
    if entry.get("ability_score"):
        prefix.append(f"Ability: {_name_of(entry['ability_score'])}")
    if entry.get("races") and not owners:
        prefix.append(f"Races: {_names(entry['races'])}")

    return ("\n".join(prefix) + "\n\n" + body).strip() if prefix else body.strip()


RENDERERS = {
    "Monsters": render_monster,
    "Spells": render_spell,
    "Equipment": render_equipment,
    "Magic-Items": render_magic_item,
    "Classes": render_class,
    "Races": render_race,
    "Levels": render_level,
}


def entry_title(stem: str, entry: Dict) -> str:
    """A human-readable name, including for entries that have none of their own."""
    name = entry.get("name")
    if name:
        return str(name)
    if stem == "Levels":
        return f"{_name_of(entry.get('class'))} level {entry.get('level')}"
    return str(entry.get("index", "Unknown"))


def chunk_entry(text: str, title: str, chunk_size: int, overlap: int) -> List[str]:
    """Split one entry, re-heading every piece with its title.

    Chroma embeds `page_content` and never metadata, so without this a rule
    section split into four chunks leaves three that cannot be found by name.
    """
    header = f"# {title}\n"
    if len(text) + len(header) <= chunk_size:
        return [header + text]

    from src.data.processing import split_text

    body_size = max(chunk_size - len(header), chunk_size // 2)
    return [header + piece for piece in split_text(text, body_size, overlap)]


def merge_shared_entries(entries: List[Dict], stem: str) -> List[Dict]:
    """Collapse entries that are the same text attached to different classes.

    `Ability Score Improvement` is stored once per class per level — 50 identical
    copies. Indexed separately they crowd retrieval: a query about a goblin's
    armour class came back with three copies of `Fighting Style: Defense` in the
    top six. Merged, one document says which classes it belongs to.
    """
    if stem not in {"Features", "Traits"}:
        return entries

    merged: Dict[tuple, Dict] = {}
    order: List[tuple] = []

    for entry in entries:
        desc = entry.get("desc")
        key = (entry.get("name", ""), json.dumps(desc, sort_keys=True))

        if key not in merged:
            merged[key] = dict(entry)
            merged[key]["_owners"] = []
            order.append(key)

        owner = _name_of(entry.get("class")) or _names(entry.get("races"))
        if owner and owner not in merged[key]["_owners"]:
            merged[key]["_owners"].append(owner)

    return [merged[key] for key in order]


def load_srd_documents(
    directory: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """Read the vendored SRD JSON and return chunked, labelled documents."""
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(
            f"No SRD corpus at {directory!r}. It ships with the repository under "
            f"corpus/srd — see corpus/README.md."
        )

    documents: List[Document] = []

    for stem, category in SRD_FILES.items():
        path = root / f"{stem}.json"
        if not path.is_file():
            logger.warning("SRD file missing, skipping: %s", path)
            continue

        entries = json.loads(path.read_text())
        if isinstance(entries, dict):
            entries = [entries]
        entries = merge_shared_entries(entries, stem)

        render = RENDERERS.get(stem, render_prose)
        for entry in entries:
            title = entry_title(stem, entry)
            body = render(entry).strip()
            if not body:
                continue

            metadata = {
                "source": SRD_SOURCE,
                "category": category,
                "name": title,
                # `book` and `page_number` are what the PDF path produces. Keeping
                # `book` populated means one citation function serves both.
                "book": f"{SRD_SOURCE} ({category})",
            }
            if stem == "Monsters":
                metadata["cr"] = str(entry.get("challenge_rating", ""))
                metadata["type"] = str(entry.get("type", ""))
            if stem == "Spells":
                metadata["level"] = str(entry.get("level", ""))
                metadata["school"] = _name_of(entry.get("school"))

            for piece in chunk_entry(body, title, chunk_size, chunk_overlap):
                documents.append(Document(page_content=piece, metadata=dict(metadata)))

        logger.info("Loaded %s", category)

    return documents
