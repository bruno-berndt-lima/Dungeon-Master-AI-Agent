"""The d20 rolls: ability checks, saving throws, attack rolls.

Every function takes an optional `rng` (anything with `randint`) and returns a
frozen result that says exactly what was rolled and why the number came out
as it did. The result's `describe()` is what the Dungeon Master narrates from,
so it has to be complete: dice shown, modifier shown, DC shown, verdict shown.

Rules applied, per SRD 5.1:

- Advantage rolls two d20 and keeps the higher; disadvantage keeps the lower.
  Any advantage and any disadvantage together cancel to a normal roll, no
  matter how many of each (`resolve_mode`).
- A natural 20 on an **attack** always hits and is a critical hit; a natural 1
  always misses. Neither is special on an ability check or a saving throw.
- A critical hit rolls the weapon's damage dice twice; the ability modifier
  is added once.
- A check or save with a DC succeeds when the total *equals or exceeds* it.
"""

import random
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict

from src.engine.character import (
    ABILITY_NAMES,
    SKILL_ABILITY,
    Ability,
    Character,
    Skill,
    Weapon,
)
from src.utils.dice import DiceRoller, RandomSource


class RollMode(str, Enum):
    NORMAL = "normal"
    ADVANTAGE = "advantage"
    DISADVANTAGE = "disadvantage"


def resolve_mode(*modes: RollMode) -> RollMode:
    """Combine sources of advantage and disadvantage the way the rules do."""
    has_advantage = RollMode.ADVANTAGE in modes
    has_disadvantage = RollMode.DISADVANTAGE in modes
    if has_advantage and has_disadvantage:
        return RollMode.NORMAL
    if has_advantage:
        return RollMode.ADVANTAGE
    if has_disadvantage:
        return RollMode.DISADVANTAGE
    return RollMode.NORMAL


def roll_d20(mode: RollMode = RollMode.NORMAL, rng: Optional[RandomSource] = None) -> Tuple[int, Tuple[int, ...]]:
    """Returns (the die that counts, every die rolled)."""
    source = rng if rng is not None else random
    mode = RollMode(mode)
    if mode == RollMode.NORMAL:
        roll = source.randint(1, 20)
        return roll, (roll,)
    first, second = source.randint(1, 20), source.randint(1, 20)
    kept = max(first, second) if mode == RollMode.ADVANTAGE else min(first, second)
    return kept, (first, second)


@dataclass(frozen=True)
class CheckResult:
    kind: str  # "check" | "save" | "attack"
    label: str  # "Dexterity (Stealth)", "Constitution save", "Longsword attack"
    mode: RollMode
    rolls: Tuple[int, ...]
    kept: int
    modifier: int
    total: int
    dc: Optional[int] = None
    success: Optional[bool] = None

    @property
    def natural_20(self) -> bool:
        return self.kept == 20

    @property
    def natural_1(self) -> bool:
        return self.kept == 1

    def describe(self) -> str:
        dice = f"d20 {list(self.rolls)}→{self.kept}" if len(self.rolls) > 1 else f"d20 {self.kept}"
        sign = "+" if self.modifier >= 0 else "−"
        line = f"{self.label} {self.kind}: {dice} {sign} {abs(self.modifier)} = {self.total}"
        if self.mode != RollMode.NORMAL:
            line += f" ({self.mode.value})"
        if self.dc is not None:
            line += f" vs DC {self.dc} — {'success' if self.success else 'failure'}"
        return line


@dataclass(frozen=True)
class DamageRoll:
    dice: str
    rolls: Tuple[int, ...]
    modifier: int
    total: int
    damage_type: str
    critical: bool

    def describe(self) -> str:
        sign = "+" if self.modifier >= 0 else "−"
        crit = " (critical: dice doubled)" if self.critical else ""
        return f"{self.dice} {list(self.rolls)} {sign} {abs(self.modifier)} = {self.total} {self.damage_type}{crit}"


@dataclass(frozen=True)
class AttackResult:
    roll: CheckResult
    weapon: str
    target_ac: int
    hit: bool
    critical: bool
    damage: Optional[DamageRoll]  # None on a miss

    def describe(self) -> str:
        outcome = "critical hit" if self.critical else ("hit" if self.hit else "miss")
        line = f"{self.roll.describe()} vs AC {self.target_ac} — {outcome}"
        if self.damage:
            line += f"; damage {self.damage.describe()}"
        return line


def _resolve(kind: str, label: str, modifier: int, dc: Optional[int], mode: RollMode, rng) -> CheckResult:
    mode = RollMode(mode)
    kept, rolls = roll_d20(mode, rng)
    total = kept + modifier
    success = None if dc is None else total >= dc
    return CheckResult(kind, label, mode, rolls, kept, modifier, total, dc, success)


def ability_check(
    character: Character,
    ability: Ability,
    skill: Optional[Skill] = None,
    dc: Optional[int] = None,
    mode: RollMode = RollMode.NORMAL,
    rng: Optional[RandomSource] = None,
) -> CheckResult:
    """A skill check when `skill` is given, a raw ability check otherwise.

    The skill's own ability wins over `ability` if they disagree — the caller
    named the skill, and Stealth is Dexterity no matter what was passed.
    """
    if skill is not None:
        skill = Skill(skill)
        ability = SKILL_ABILITY[skill]
        modifier = character.skill_modifier(skill)
        label = f"{ABILITY_NAMES[ability]} ({skill.value})"
    else:
        ability = Ability(ability)
        modifier = character.ability_modifier(ability)
        label = ABILITY_NAMES[ability]
    return _resolve("check", label, modifier, dc, mode, rng)


def saving_throw(
    character: Character,
    ability: Ability,
    dc: int,
    mode: RollMode = RollMode.NORMAL,
    rng: Optional[RandomSource] = None,
) -> CheckResult:
    ability = Ability(ability)
    return _resolve(
        "save", f"{ABILITY_NAMES[ability]}", character.save_modifier(ability), dc, mode, rng
    )


def _doubled(notation: str) -> str:
    """"1d8+1d6" → "2d8+2d6": a critical hit rolls every damage die twice."""
    return "+".join(f"{q * 2}d{s}" for q, s in DiceRoller.parse_dice_string(notation))


def roll_damage(
    weapon: Weapon,
    modifier: int,
    critical: bool = False,
    two_handed: bool = False,
    rng: Optional[RandomSource] = None,
) -> DamageRoll:
    dice = weapon.versatile_dice if (two_handed and weapon.versatile_dice) else weapon.damage_dice
    notation = _doubled(dice) if critical else dice
    rolled = DiceRoller.roll_multiple(notation, rng)
    rolls = tuple(r for group in rolled for r in group.results)
    total = max(0, sum(rolls) + modifier)  # damage never heals
    return DamageRoll(notation, rolls, modifier, total, weapon.damage_type, critical)


def attack_roll(
    character: Character,
    weapon: Weapon,
    target_ac: int,
    mode: RollMode = RollMode.NORMAL,
    two_handed: bool = False,
    rng: Optional[RandomSource] = None,
) -> AttackResult:
    """One weapon attack against an Armor Class, damage included on a hit."""
    roll = _resolve("attack", weapon.name, character.attack_bonus(weapon), None, mode, rng)

    if roll.natural_1:
        hit, critical = False, False
    elif roll.natural_20:
        hit, critical = True, True
    else:
        hit, critical = roll.total >= target_ac, False

    damage = None
    if hit:
        damage = roll_damage(weapon, character.damage_bonus(weapon), critical, two_handed, rng)

    return AttackResult(roll, weapon.name, target_ac, hit, critical, damage)


class PendingCheck(BaseModel):
    """A roll the Dungeon Master has asked a player for and is waiting on.

    Lives in `GameState["pending"]` between the DM's request and the player's
    `/roll` (ROADMAP §4). The DC is kept here so the player never sees it.
    """

    model_config = ConfigDict(frozen=True)

    player: str  # character name
    kind: Literal["check", "save"]
    ability: Ability
    skill: Optional[Skill] = None
    dc: int
    mode: RollMode = RollMode.NORMAL
    reason: str = ""

    @property
    def label(self) -> str:
        ability = ABILITY_NAMES[Ability(self.ability)]
        if self.kind == "save":
            return f"{ability} saving throw"
        if self.skill:
            return f"{ability} ({Skill(self.skill).value}) check"
        return f"{ability} check"

    def resolve(self, character: Character, rng: Optional[RandomSource] = None) -> CheckResult:
        if self.kind == "save":
            return saving_throw(character, self.ability, self.dc, self.mode, rng)
        return ability_check(character, self.ability, self.skill, self.dc, self.mode, rng)
