"""Structured access to the vendored SRD 5.1 JSON in `corpus/srd/`.

The corpus was indexed for retrieval in PR-09; this package reads the same
files as *data*. A monster's AC, HP, attack bonus and damage dice are fields
in `Monsters.json`, so the combat engine looks them up rather than asking a
model to remember them.

    from src.srd import monster, spell, equipment, condition
    monster("goblin")["armor_class"]      # exact, case-insensitive, or by slug
    monster("gobln")                      # within edit distance 2 still resolves
    monster("beholder")                   # UnknownEntry, with suggestions

`src.srd.bestiary.summon("goblin")` turns a stat block into an engine
`Combatant`.
"""

from src.srd.data import (  # noqa: F401
    UnknownEntry,
    condition,
    equipment,
    find,
    magic_item,
    monster,
    names,
    spell,
)
