"""Read `corpus/srd/*.json` once, look entries up by name.

Lookup order: exact index (`"adult-red-dragon"`), exact name (case-
insensitive), then the closest name within `FUZZY_DISTANCE` edits — a player
typing `gobln` should still get the goblin. Ties are broken by similarity,
which is what keeps `ogr` resolving to the ogre rather than the orc. Anything
further away raises `UnknownEntry` naming the nearest candidates, because a
silently wrong monster is worse than a question.
"""

import difflib
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

from src.config import SRD_DIRECTORY

FUZZY_DISTANCE = 2

# File stem -> what to call one entry in an error message.
CATEGORIES = {
    "Monsters": "monster",
    "Spells": "spell",
    "Equipment": "equipment",
    "Conditions": "condition",
    "Magic-Items": "magic item",
}


class UnknownEntry(KeyError):
    """No entry by that name, and nothing close enough to assume."""


def _slug(text: str) -> str:
    return text.strip().lower().replace(" ", "-")


@lru_cache(maxsize=None)
def _entries(stem: str) -> Tuple[dict, ...]:
    path = Path(SRD_DIRECTORY) / f"{stem}.json"
    if not path.is_file():
        raise FileNotFoundError(f"No SRD file at {path}; the corpus ships under corpus/srd.")
    data = json.loads(path.read_text())
    return tuple(data if isinstance(data, list) else [data])


@lru_cache(maxsize=None)
def _by_key(stem: str) -> Dict[str, dict]:
    """Every entry under both its index and its lowercased name."""
    table: Dict[str, dict] = {}
    for entry in _entries(stem):
        table.setdefault(str(entry.get("index", "")).lower(), entry)
        table.setdefault(str(entry.get("name", "")).strip().lower(), entry)
    table.pop("", None)
    return table


def names(stem: str) -> List[str]:
    """Display names in file order — `names("Monsters")` is the bestiary."""
    return [str(e.get("name", "")) for e in _entries(stem)]


def levenshtein(a: str, b: str, limit: int = FUZZY_DISTANCE) -> int:
    """Edit distance, giving up early (returning limit + 1) once it is exceeded."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        if min(current) > limit:
            return limit + 1
        previous = current
    return previous[-1]


def find(stem: str, name: str) -> dict:
    """The entry for `name` in the file `stem`, exactly or nearly."""
    key = (name or "").strip().lower()
    if not key:
        raise UnknownEntry(f"No {CATEGORIES.get(stem, stem)} name given.")

    table = _by_key(stem)
    for candidate in (key, _slug(key)):
        if candidate in table:
            return table[candidate]

    display = [n for n in names(stem) if n]
    scored = [
        (levenshtein(key, n.lower()), -difflib.SequenceMatcher(None, key, n.lower()).ratio(), n)
        for n in display
    ]
    close = sorted(s for s in scored if s[0] <= FUZZY_DISTANCE)
    if close:
        return table[close[0][2].lower()]

    suggestions = difflib.get_close_matches(key, [n.lower() for n in display], n=3, cutoff=0.5)
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    raise UnknownEntry(f"No {CATEGORIES.get(stem, stem)} called {name!r}.{hint}")


def monster(name: str) -> dict:
    return find("Monsters", name)


def spell(name: str) -> dict:
    return find("Spells", name)


def equipment(name: str) -> dict:
    return find("Equipment", name)


def condition(name: str) -> dict:
    return find("Conditions", name)


def magic_item(name: str) -> dict:
    return find("Magic-Items", name)
