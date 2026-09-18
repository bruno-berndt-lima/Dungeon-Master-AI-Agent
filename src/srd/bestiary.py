"""Monsters as engine combatants."""

from typing import List, Optional

from src.engine.combatant import Combatant
from src.srd.data import monster
from src.utils.dice import RandomSource


def summon(
    name: str,
    combatant_id: Optional[str] = None,
    rng: Optional[RandomSource] = None,
    roll_hp: bool = False,
) -> Combatant:
    """One monster from the SRD, ready for initiative."""
    return Combatant.from_monster(monster(name), combatant_id=combatant_id, rng=rng, roll_hp=roll_hp)


def summon_group(
    name: str,
    count: int,
    rng: Optional[RandomSource] = None,
    roll_hp: bool = False,
) -> List[Combatant]:
    """`count` of the same monster, ids `goblin-1`, `goblin-2`, ..."""
    entry = monster(name)
    return [
        Combatant.from_monster(entry, combatant_id=f"{entry['index']}-{i}", rng=rng, roll_hp=roll_hp)
        for i in range(1, count + 1)
    ]
