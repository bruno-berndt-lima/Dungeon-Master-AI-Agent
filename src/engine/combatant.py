"""A creature in a fight, built from an SRD stat block.

`Combatant` is the monster-side counterpart of `Character`: the numbers an
encounter needs, frozen, JSON round-trippable, with the durable text (traits,
multiattack, legendary actions) kept as strings for the Dungeon Master to
narrate from. `from_monster` reads the shapes `Monsters.json` actually uses,
three of which are easy to get wrong:

- damage is a *list* of components — an adult black dragon's bite is
  `2d10+6 piercing` plus `1d8 acid` — and a few weapons offer a *choice* (a
  djinni's scimitar is lightning or thunder; the first option is taken).
  Riders that need a saving throw (a giant spider's poison) are not damage
  entries at all; they stay in the attack's `desc` for the DM to adjudicate;
- some damage is a flat number with no dice at all (a rat's bite: `"1"`);
- AC is a list of typed entries (`natural`, `dex`, `armor`); the first value
  is the creature's AC.
"""

import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.engine.character import (
    SKILL_ABILITY,
    Abilities,
    Ability,
    Character,
    Condition,
    Skill,
    ability_modifier,
)
from src.utils.dice import DiceRoller, RandomSource

# "1d6+2", "2d8", "3d6 + 4", "1d4-1" — dice with an optional flat bonus.
DICE_WITH_BONUS = re.compile(
    r"^\s*(?P<dice>\d*d\d+(?:\s*\+\s*\d*d\d+)*)(?:\s*(?P<sign>[+-])\s*(?P<flat>\d+))?\s*$",
    re.IGNORECASE,
)
# "1" — a rat's bite. No dice at all.
FLAT_ONLY = re.compile(r"^\s*(?P<flat>\d+)\s*$")


class Damage(BaseModel):
    """One component of an attack's damage: `1d6+2 slashing`, or a flat `1`."""

    model_config = ConfigDict(frozen=True)

    dice: Optional[str] = None  # None for flat damage
    bonus: int = 0
    damage_type: str

    @classmethod
    def parse(cls, notation: str, damage_type: str) -> "Damage":
        text = notation or ""
        flat_only = FLAT_ONLY.match(text)
        if flat_only:
            return cls(dice=None, bonus=int(flat_only.group("flat")), damage_type=damage_type)

        match = DICE_WITH_BONUS.match(text)
        if not match:
            raise ValueError(f"unreadable damage notation {notation!r}")
        dice = match.group("dice").replace(" ", "")
        DiceRoller.parse_dice_string(dice)  # raises on junk like "0d6"
        bonus = int(match.group("flat") or 0)
        if match.group("sign") == "-":
            bonus = -bonus
        return cls(dice=dice, bonus=bonus, damage_type=damage_type)

    @property
    def notation(self) -> str:
        if self.dice is None:
            return str(self.bonus)
        if self.bonus == 0:
            return self.dice
        return f"{self.dice}{'+' if self.bonus > 0 else '-'}{abs(self.bonus)}"

    @property
    def average(self) -> int:
        total = self.bonus
        if self.dice:
            for quantity, sides in DiceRoller.parse_dice_string(self.dice):
                total += quantity * (sides + 1) / 2
        return int(total)


class Attack(BaseModel):
    """An action with an attack bonus. `desc` keeps riders (a poison save)."""

    model_config = ConfigDict(frozen=True)

    name: str
    attack_bonus: int
    damage: List[Damage] = Field(default_factory=list)
    ranged: bool = False
    desc: str = ""

    def describe(self) -> str:
        parts = ", ".join(f"{d.notation} {d.damage_type}" for d in self.damage) or "no damage"
        return f"{self.name} ({'ranged' if self.ranged else 'melee'}, +{self.attack_bonus} to hit, {parts})"


def _first_damage_options(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten a damage list, taking the first option of any 'choose' set."""
    flat: List[Dict[str, Any]] = []
    for item in entry.get("damage") or []:
        if not isinstance(item, dict):
            continue
        if "damage_dice" in item:
            flat.append(item)
        elif "from" in item:
            options = (item.get("from") or {}).get("options") or []
            if options and isinstance(options[0], dict) and "damage_dice" in options[0]:
                flat.append(options[0])
    return flat


def _armor_class(value: Any) -> int:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("value"), int):
                return item["value"]
    if isinstance(value, int):
        return value
    return 10


def _name_of(value: Any) -> str:
    return str(value.get("name", "")) if isinstance(value, dict) else str(value or "")


def _block(entries: Optional[List[Dict[str, Any]]]) -> List[str]:
    return [
        f"{e.get('name', '').strip()}. {e.get('desc', '').strip()}".strip(". ")
        for e in entries or []
        if isinstance(e, dict) and (e.get("name") or e.get("desc"))
    ]


class Combatant(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    kind: Literal["monster", "character"] = "monster"
    player_id: Optional[str] = None

    armor_class: int = Field(ge=0)
    max_hp: int = Field(ge=1)
    current_hp: int = Field(ge=0)
    temp_hp: int = Field(0, ge=0)
    abilities: Abilities = Field(default_factory=Abilities)
    save_bonuses: Dict[Ability, int] = Field(default_factory=dict)  # listed totals
    skill_bonuses: Dict[str, int] = Field(default_factory=dict)
    speed: Dict[str, str] = Field(default_factory=dict)

    size: str = ""
    creature_type: str = ""
    alignment: str = ""
    challenge_rating: float = 0
    xp: int = 0

    attacks: List[Attack] = Field(default_factory=list)
    multiattack: Optional[str] = None
    traits: List[str] = Field(default_factory=list)
    legendary_actions: List[str] = Field(default_factory=list)
    damage_vulnerabilities: List[str] = Field(default_factory=list)
    damage_resistances: List[str] = Field(default_factory=list)
    damage_immunities: List[str] = Field(default_factory=list)
    condition_immunities: List[str] = Field(default_factory=list)
    senses: Dict[str, str] = Field(default_factory=dict)
    languages: str = ""
    conditions: List[Condition] = Field(default_factory=list)

    # Dying, for characters. Monsters simply die at 0 HP.
    dead: bool = False
    death_successes: int = Field(0, ge=0, le=3)
    death_failures: int = Field(0, ge=0, le=3)
    stable: bool = False

    @model_validator(mode="before")
    @classmethod
    def _fill_defaults(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if data.get("current_hp") is None and "max_hp" in data:
                data["current_hp"] = data["max_hp"]
        return data

    @model_validator(mode="after")
    def _within_bounds(self) -> "Combatant":
        if self.current_hp > self.max_hp:
            raise ValueError(f"current_hp {self.current_hp} exceeds max_hp {self.max_hp}")
        return self

    # --- building one from the SRD ------------------------------------------

    @classmethod
    def from_monster(
        cls,
        entry: Dict[str, Any],
        combatant_id: Optional[str] = None,
        rng: Optional[RandomSource] = None,
        roll_hp: bool = False,
    ) -> "Combatant":
        """A stat block as `Monsters.json` stores it → a combatant.

        HP is the listed average unless `roll_hp`, in which case the hit dice
        are rolled and the Constitution modifier added per die, which is what
        `hit_points_roll` (`7d10+21` for an ogre) spells out.
        """
        abilities = Abilities(
            strength=entry.get("strength", 10),
            dexterity=entry.get("dexterity", 10),
            constitution=entry.get("constitution", 10),
            intelligence=entry.get("intelligence", 10),
            wisdom=entry.get("wisdom", 10),
            charisma=entry.get("charisma", 10),
        )

        max_hp = int(entry.get("hit_points") or 1)
        if roll_hp and entry.get("hit_dice"):
            rolled = DiceRoller.roll_multiple(str(entry["hit_dice"]), rng)
            dice_count = sum(len(r.results) for r in rolled)
            max_hp = max(1, sum(r.total for r in rolled) + dice_count * ability_modifier(abilities.constitution))

        save_bonuses: Dict[Ability, int] = {}
        skill_bonuses: Dict[str, int] = {}
        for prof in entry.get("proficiencies") or []:
            label = _name_of(prof.get("proficiency"))
            value = int(prof.get("value", 0))
            if label.startswith("Saving Throw:"):
                save_bonuses[Ability(label.split(":")[1].strip().upper())] = value
            elif label.startswith("Skill:"):
                skill_bonuses[label.split(":")[1].strip()] = value

        attacks: List[Attack] = []
        multiattack = None
        for action in entry.get("actions") or []:
            if not isinstance(action, dict):
                continue
            if action.get("name") == "Multiattack":
                multiattack = str(action.get("desc", "")).strip()
                continue
            if action.get("attack_bonus") is None:
                continue
            damage = [
                Damage.parse(str(d["damage_dice"]), _name_of(d.get("damage_type")).lower())
                for d in _first_damage_options(action)
            ]
            desc = str(action.get("desc", "")).strip()
            attacks.append(
                Attack(
                    name=str(action.get("name", "")),
                    attack_bonus=int(action["attack_bonus"]),
                    damage=damage,
                    ranged=desc.lower().startswith("ranged"),
                    desc=desc,
                )
            )

        def _strings(key: str) -> List[str]:
            value = entry.get(key) or []
            return [str(v) for v in value] if isinstance(value, list) else [str(value)]

        index = str(entry.get("index") or entry.get("name", "monster")).lower()
        return cls(
            id=combatant_id or index,
            name=str(entry.get("name", index)),
            armor_class=_armor_class(entry.get("armor_class")),
            max_hp=max_hp,
            abilities=abilities,
            save_bonuses=save_bonuses,
            skill_bonuses=skill_bonuses,
            speed={k: str(v) for k, v in (entry.get("speed") or {}).items()},
            size=str(entry.get("size", "")),
            creature_type=str(entry.get("type", "")),
            alignment=str(entry.get("alignment", "")),
            challenge_rating=float(entry.get("challenge_rating") or 0),
            xp=int(entry.get("xp") or 0),
            attacks=attacks,
            multiattack=multiattack,
            traits=_block(entry.get("special_abilities")),
            legendary_actions=_block(entry.get("legendary_actions")),
            damage_vulnerabilities=_strings("damage_vulnerabilities"),
            damage_resistances=_strings("damage_resistances"),
            damage_immunities=_strings("damage_immunities"),
            condition_immunities=[_name_of(c) for c in entry.get("condition_immunities") or []],
            senses={k: str(v) for k, v in (entry.get("senses") or {}).items()},
            languages=str(entry.get("languages", "")),
        )

    @classmethod
    def from_character(cls, character: Character) -> "Combatant":
        """A player character as it takes part in a fight.

        The sheet stays the source of truth between fights; `combat.sync_party`
        writes HP, temporary HP, conditions and death back when one ends.
        Listed saves and skills are filled for every ability and skill so the
        two kinds of combatant answer the same questions the same way.
        """
        attacks = [
            Attack(
                name=weapon.name,
                attack_bonus=character.attack_bonus(weapon),
                damage=[Damage(dice=weapon.damage_dice, bonus=character.damage_bonus(weapon), damage_type=weapon.damage_type)],
                ranged=weapon.ranged,
                desc=f"{'Ranged' if weapon.ranged else 'Melee'} Weapon Attack with a {weapon.name.lower()}.",
            )
            for weapon in character.weapons
        ]
        return cls(
            id=character.name,
            name=character.name,
            kind="character",
            player_id=character.player_id,
            armor_class=character.armor_class,
            max_hp=character.max_hp,
            current_hp=character.current_hp,
            temp_hp=character.temp_hp,
            abilities=character.abilities,
            save_bonuses={a: character.save_modifier(a) for a in Ability},
            skill_bonuses={s.value: character.skill_modifier(s) for s in Skill},
            speed={"walk": f"{character.speed} ft."},
            size="Medium",
            creature_type=f"humanoid ({character.race.lower()})",
            alignment="",
            attacks=attacks,
            conditions=list(character.conditions),
            dead=character.dead,
        )

    # --- the numbers a fight asks for ---------------------------------------

    def ability_modifier(self, ability: Ability) -> int:
        return self.abilities.modifier(ability)

    def save_modifier(self, ability: Ability) -> int:
        """A listed save is a total; otherwise the ability modifier."""
        ability = Ability(ability)
        return self.save_bonuses.get(ability, self.ability_modifier(ability))

    def skill_modifier(self, skill: str) -> int:
        """Listed skills are totals; anything else is unproficient — the
        ability is looked up through the character module's table."""
        for listed, value in self.skill_bonuses.items():
            if listed.lower() == str(skill).lower():
                return value
        return self.ability_modifier(SKILL_ABILITY[Skill(skill)])

    @property
    def initiative_modifier(self) -> int:
        return self.ability_modifier(Ability.DEX)

    @property
    def is_alive(self) -> bool:
        return not self.dead

    @property
    def is_conscious(self) -> bool:
        return not self.dead and self.current_hp > 0 and not self.has(Condition.UNCONSCIOUS)

    def has(self, condition: Condition) -> bool:
        return Condition(condition) in self.conditions

    def attack(self, name: str) -> Attack:
        for attack in self.attacks:
            if attack.name.lower() == name.lower():
                return attack
        available = ", ".join(a.name for a in self.attacks) or "nothing"
        raise KeyError(f"{self.name} has no attack called {name!r} (has: {available})")

    def sheet_line(self) -> str:
        hp = f"{self.current_hp}/{self.max_hp} HP"
        if self.temp_hp:
            hp += f" (+{self.temp_hp} temp)"
        conditions = f", {', '.join(c.value for c in self.conditions)}" if self.conditions else ""
        if self.dead:
            conditions += ", dead"
        elif self.current_hp == 0 and self.kind == "character":
            conditions += f", dying ({self.death_successes}✓ {self.death_failures}✗{', stable' if self.stable else ''})"
        who = self.id if self.id == self.name else f"{self.id} ({self.name})"
        return f"{who}: AC {self.armor_class}, {hp}{conditions}"
