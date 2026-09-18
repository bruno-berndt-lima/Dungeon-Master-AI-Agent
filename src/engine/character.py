"""Character sheets.

A `Character` is a pydantic model so that it round-trips through JSON
unchanged — it lives inside the checkpointed `GameState`, and later inside
whatever the Discord bot stores. Everything derivable is a method, never a
stored field, so a sheet cannot disagree with itself: the proficiency bonus
comes from the level, a skill modifier from the ability and the proficiency,
passive Perception from the Perception modifier.

Rules covered here are the ones a level-1 party needs. Multiclassing, feats,
and spellcasting are deliberately absent (ROADMAP §5, PR-24).
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.utils.dice import DiceRoller


class Ability(str, Enum):
    STR = "STR"
    DEX = "DEX"
    CON = "CON"
    INT = "INT"
    WIS = "WIS"
    CHA = "CHA"


ABILITY_NAMES = {
    Ability.STR: "Strength",
    Ability.DEX: "Dexterity",
    Ability.CON: "Constitution",
    Ability.INT: "Intelligence",
    Ability.WIS: "Wisdom",
    Ability.CHA: "Charisma",
}


class Skill(str, Enum):
    ACROBATICS = "Acrobatics"
    ANIMAL_HANDLING = "Animal Handling"
    ARCANA = "Arcana"
    ATHLETICS = "Athletics"
    DECEPTION = "Deception"
    HISTORY = "History"
    INSIGHT = "Insight"
    INTIMIDATION = "Intimidation"
    INVESTIGATION = "Investigation"
    MEDICINE = "Medicine"
    NATURE = "Nature"
    PERCEPTION = "Perception"
    PERFORMANCE = "Performance"
    PERSUASION = "Persuasion"
    RELIGION = "Religion"
    SLEIGHT_OF_HAND = "Sleight of Hand"
    STEALTH = "Stealth"
    SURVIVAL = "Survival"


# SRD 5.1, "Using Ability Scores": which ability each skill rides on.
SKILL_ABILITY: Dict[Skill, Ability] = {
    Skill.ATHLETICS: Ability.STR,
    Skill.ACROBATICS: Ability.DEX,
    Skill.SLEIGHT_OF_HAND: Ability.DEX,
    Skill.STEALTH: Ability.DEX,
    Skill.ARCANA: Ability.INT,
    Skill.HISTORY: Ability.INT,
    Skill.INVESTIGATION: Ability.INT,
    Skill.NATURE: Ability.INT,
    Skill.RELIGION: Ability.INT,
    Skill.ANIMAL_HANDLING: Ability.WIS,
    Skill.INSIGHT: Ability.WIS,
    Skill.MEDICINE: Ability.WIS,
    Skill.PERCEPTION: Ability.WIS,
    Skill.SURVIVAL: Ability.WIS,
    Skill.DECEPTION: Ability.CHA,
    Skill.INTIMIDATION: Ability.CHA,
    Skill.PERFORMANCE: Ability.CHA,
    Skill.PERSUASION: Ability.CHA,
}


class Proficiency(str, Enum):
    PROFICIENT = "proficient"
    EXPERTISE = "expertise"  # double proficiency bonus (rogue, bard)


class Condition(str, Enum):
    """SRD 5.1 Appendix A. Their mechanical effects are applied in `combat.py`."""

    BLINDED = "blinded"
    CHARMED = "charmed"
    DEAFENED = "deafened"
    FRIGHTENED = "frightened"
    GRAPPLED = "grappled"
    INCAPACITATED = "incapacitated"
    INVISIBLE = "invisible"
    PARALYZED = "paralyzed"
    PETRIFIED = "petrified"
    POISONED = "poisoned"
    PRONE = "prone"
    RESTRAINED = "restrained"
    STUNNED = "stunned"
    UNCONSCIOUS = "unconscious"


# `Abilities` field per ability, so the model can be addressed by enum.
ABILITY_FIELDS = {
    Ability.STR: "strength",
    Ability.DEX: "dexterity",
    Ability.CON: "constitution",
    Ability.INT: "intelligence",
    Ability.WIS: "wisdom",
    Ability.CHA: "charisma",
}


def ability_modifier(score: int) -> int:
    """SRD: (score − 10) ÷ 2, rounded down. Floor division gets 9 → −1 right."""
    return (score - 10) // 2


def proficiency_bonus_for(level: int) -> int:
    """+2 at levels 1–4, +3 at 5–8, +4 at 9–12, +5 at 13–16, +6 at 17–20."""
    return 2 + (level - 1) // 4


class Abilities(BaseModel):
    model_config = ConfigDict(frozen=True)

    strength: int = Field(10, ge=1, le=30)
    dexterity: int = Field(10, ge=1, le=30)
    constitution: int = Field(10, ge=1, le=30)
    intelligence: int = Field(10, ge=1, le=30)
    wisdom: int = Field(10, ge=1, le=30)
    charisma: int = Field(10, ge=1, le=30)

    def score(self, ability: Ability) -> int:
        return getattr(self, ABILITY_FIELDS[Ability(ability)])

    def modifier(self, ability: Ability) -> int:
        return ability_modifier(self.score(ability))


class Weapon(BaseModel):
    """Enough of a weapon to make an attack roll with it."""

    model_config = ConfigDict(frozen=True)

    name: str
    damage_dice: str  # "1d8"
    damage_type: str  # "slashing"
    finesse: bool = False
    ranged: bool = False
    thrown: bool = False
    light: bool = False
    heavy: bool = False
    two_handed: bool = False
    versatile_dice: Optional[str] = None  # damage when wielded two-handed

    @field_validator("damage_dice", "versatile_dice")
    @classmethod
    def _rollable(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            DiceRoller.parse_dice_string(value)  # raises on junk
        return value


class Character(BaseModel):
    """One player's sheet. Immutable in use: change it with `model_copy(update=...)`."""

    model_config = ConfigDict(frozen=True)

    player_id: str
    name: str
    race: str
    character_class: str
    level: int = Field(1, ge=1, le=20)
    abilities: Abilities
    skills: Dict[Skill, Proficiency] = Field(default_factory=dict)
    saves: List[Ability] = Field(default_factory=list)
    armor_class: int = Field(ge=1)
    max_hp: int = Field(ge=1)
    current_hp: int = Field(ge=0)
    temp_hp: int = Field(0, ge=0)
    hit_die: int = Field(8, ge=4)  # d8: the size, not a notation
    hit_dice_remaining: int = Field(ge=0)
    speed: int = Field(30, ge=0)
    conditions: List[Condition] = Field(default_factory=list)
    inventory: List[str] = Field(default_factory=list)
    weapons: List[Weapon] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="before")
    @classmethod
    def _fill_defaults(cls, data: Any) -> Any:
        """A fresh sheet is at full health with every hit die unspent."""
        if isinstance(data, dict):
            data = dict(data)
            if data.get("current_hp") is None and "max_hp" in data:
                data["current_hp"] = data["max_hp"]
            if data.get("hit_dice_remaining") is None:
                data["hit_dice_remaining"] = data.get("level", 1)
        return data

    @model_validator(mode="after")
    def _within_bounds(self) -> "Character":
        if self.current_hp > self.max_hp:
            raise ValueError(f"current_hp {self.current_hp} exceeds max_hp {self.max_hp}")
        if self.hit_dice_remaining > self.level:
            raise ValueError("more hit dice remaining than levels")
        return self

    # --- derived numbers ----------------------------------------------------

    @property
    def proficiency_bonus(self) -> int:
        return proficiency_bonus_for(self.level)

    def ability_modifier(self, ability: Ability) -> int:
        return self.abilities.modifier(ability)

    def skill_modifier(self, skill: Skill) -> int:
        skill = Skill(skill)
        bonus = self.ability_modifier(SKILL_ABILITY[skill])
        proficiency = self.skills.get(skill)
        if proficiency == Proficiency.PROFICIENT:
            bonus += self.proficiency_bonus
        elif proficiency == Proficiency.EXPERTISE:
            bonus += 2 * self.proficiency_bonus
        return bonus

    def save_modifier(self, ability: Ability) -> int:
        ability = Ability(ability)
        bonus = self.ability_modifier(ability)
        if ability in self.saves:
            bonus += self.proficiency_bonus
        return bonus

    @property
    def passive_perception(self) -> int:
        return 10 + self.skill_modifier(Skill.PERCEPTION)

    @property
    def initiative_modifier(self) -> int:
        return self.ability_modifier(Ability.DEX)

    def attack_ability(self, weapon: Weapon) -> Ability:
        """Ranged → DEX. Finesse → whichever of STR/DEX is higher. Else STR.

        A thrown weapon without finesse (javelin, handaxe) still uses STR.
        """
        if weapon.ranged:
            return Ability.DEX
        if weapon.finesse:
            str_mod = self.ability_modifier(Ability.STR)
            dex_mod = self.ability_modifier(Ability.DEX)
            return Ability.DEX if dex_mod > str_mod else Ability.STR
        return Ability.STR

    def attack_bonus(self, weapon: Weapon) -> int:
        """Ability modifier plus proficiency. Every pregen is proficient with
        what it carries; per-weapon proficiency is not modelled."""
        return self.ability_modifier(self.attack_ability(weapon)) + self.proficiency_bonus

    def damage_bonus(self, weapon: Weapon) -> int:
        return self.ability_modifier(self.attack_ability(weapon))

    # --- state queries ------------------------------------------------------

    def has(self, condition: Condition) -> bool:
        return Condition(condition) in self.conditions

    @property
    def is_conscious(self) -> bool:
        return self.current_hp > 0 and not self.has(Condition.UNCONSCIOUS)

    def weapon(self, name: str) -> Weapon:
        for weapon in self.weapons:
            if weapon.name.lower() == name.lower():
                return weapon
        carried = ", ".join(w.name for w in self.weapons) or "nothing"
        raise KeyError(f"{self.name} carries no weapon called {name!r} (carries: {carried})")

    def sheet_line(self) -> str:
        """One line for the DM's scene sheet."""
        hp = f"{self.current_hp}/{self.max_hp} HP"
        if self.temp_hp:
            hp += f" (+{self.temp_hp} temp)"
        conditions = f", {', '.join(c.value for c in self.conditions)}" if self.conditions else ""
        return (
            f"{self.name} ({self.race} {self.character_class} {self.level}): "
            f"AC {self.armor_class}, {hp}{conditions}"
        )
