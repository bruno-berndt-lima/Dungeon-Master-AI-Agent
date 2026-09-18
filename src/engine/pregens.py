"""Six pregenerated level-1 characters, SRD content only.

Built from the standard array (15, 14, 13, 12, 10, 8) plus the SRD racial
bonuses, so every number below can be checked against the rulebook. `/join
fighter as Dorn` (ROADMAP PR-21) hands out a copy; nobody needs a character
creation flow to sit down at the table.

The templates are module constants and `pregen()` returns a *copy* with the
player's identity filled in — a template is never handed out by reference.
"""

from typing import Dict, List, Optional

from src.engine.character import (
    Abilities,
    Ability,
    Character,
    Proficiency,
    Skill,
    Weapon,
)

# SRD 5.1 weapon table, the entries the pregens carry.
WEAPONS: Dict[str, Weapon] = {
    "longsword": Weapon(name="Longsword", damage_dice="1d8", damage_type="slashing", versatile_dice="1d10"),
    "handaxe": Weapon(name="Handaxe", damage_dice="1d6", damage_type="slashing", light=True, thrown=True),
    "rapier": Weapon(name="Rapier", damage_dice="1d8", damage_type="piercing", finesse=True),
    "shortbow": Weapon(name="Shortbow", damage_dice="1d6", damage_type="piercing", ranged=True, two_handed=True),
    "dagger": Weapon(name="Dagger", damage_dice="1d4", damage_type="piercing", finesse=True, light=True, thrown=True),
    "mace": Weapon(name="Mace", damage_dice="1d6", damage_type="bludgeoning"),
    "quarterstaff": Weapon(name="Quarterstaff", damage_dice="1d6", damage_type="bludgeoning", versatile_dice="1d8"),
    "longbow": Weapon(name="Longbow", damage_dice="1d8", damage_type="piercing", ranged=True, heavy=True, two_handed=True),
    "shortsword": Weapon(name="Shortsword", damage_dice="1d6", damage_type="piercing", finesse=True, light=True),
    "greataxe": Weapon(name="Greataxe", damage_dice="1d12", damage_type="slashing", heavy=True, two_handed=True),
    "javelin": Weapon(name="Javelin", damage_dice="1d6", damage_type="piercing", thrown=True),
}

P, E = Proficiency.PROFICIENT, Proficiency.EXPERTISE

# player_id is filled by `pregen()`; the template carries a placeholder.
_TEMPLATE_PLAYER = "template"

PREGENS: Dict[str, Character] = {
    # Human +1 to every score. Chain mail (16) and a shield (+2).
    "fighter": Character(
        player_id=_TEMPLATE_PLAYER, name="Dorn Ironfist", race="Human", character_class="Fighter",
        abilities=Abilities(strength=16, dexterity=14, constitution=16, intelligence=11, wisdom=13, charisma=9),
        skills={Skill.ATHLETICS: P, Skill.PERCEPTION: P},
        saves=[Ability.STR, Ability.CON],
        armor_class=18, max_hp=13, hit_die=10, speed=30,
        weapons=[WEAPONS["longsword"], WEAPONS["handaxe"]],
        inventory=["chain mail", "shield", "backpack", "rations (5 days)", "torch"],
        notes="Fighting Style: Defense (+1 AC while armored, already counted). Second Wind 1/rest.",
    ),
    # Lightfoot halfling +2 DEX, +1 CHA. Leather armor (11 + DEX). Expertise in Stealth and Perception.
    "rogue": Character(
        player_id=_TEMPLATE_PLAYER, name="Kara Swiftfoot", race="Halfling", character_class="Rogue",
        abilities=Abilities(strength=8, dexterity=17, constitution=13, intelligence=14, wisdom=12, charisma=11),
        skills={Skill.STEALTH: E, Skill.PERCEPTION: E, Skill.SLEIGHT_OF_HAND: P, Skill.ACROBATICS: P, Skill.DECEPTION: P},
        saves=[Ability.DEX, Ability.INT],
        armor_class=14, max_hp=9, hit_die=8, speed=25,
        weapons=[WEAPONS["rapier"], WEAPONS["shortbow"], WEAPONS["dagger"]],
        inventory=["leather armor", "thieves' tools", "backpack", "20 arrows", "crowbar"],
        notes="Sneak Attack 1d6 (advantage, or an ally within 5 ft of the target). Lucky: reroll a natural 1.",
    ),
    # Hill dwarf +2 CON, +1 WIS; Dwarven Toughness +1 HP. Scale mail (14) and a shield (+2), DEX 0.
    "cleric": Character(
        player_id=_TEMPLATE_PLAYER, name="Brother Aldric", race="Dwarf", character_class="Cleric",
        abilities=Abilities(strength=13, dexterity=10, constitution=16, intelligence=8, wisdom=16, charisma=12),
        skills={Skill.MEDICINE: P, Skill.INSIGHT: P, Skill.RELIGION: P},
        saves=[Ability.WIS, Ability.CHA],
        armor_class=16, max_hp=12, hit_die=8, speed=25,
        weapons=[WEAPONS["mace"]],
        inventory=["scale mail", "shield", "holy symbol", "backpack", "healer's kit"],
        notes="Life Domain. Spells are narrated for now (ROADMAP PR-24): Cure Wounds, Bless, Sacred Flame.",
    ),
    # High elf +2 DEX, +1 INT. No armor: 10 + DEX.
    "wizard": Character(
        player_id=_TEMPLATE_PLAYER, name="Elowen Nightbreeze", race="Elf", character_class="Wizard",
        abilities=Abilities(strength=8, dexterity=16, constitution=13, intelligence=16, wisdom=12, charisma=10),
        skills={Skill.ARCANA: P, Skill.HISTORY: P, Skill.INVESTIGATION: P},
        saves=[Ability.INT, Ability.WIS],
        armor_class=13, max_hp=7, hit_die=6, speed=30,
        weapons=[WEAPONS["quarterstaff"], WEAPONS["dagger"]],
        inventory=["spellbook", "component pouch", "backpack", "ink and quill"],
        notes="Spells are narrated for now (ROADMAP PR-24): Magic Missile, Shield, Fire Bolt, Mage Hand.",
    ),
    # Wood elf +2 DEX, +1 WIS; speed 35. Leather armor (11 + DEX).
    "ranger": Character(
        player_id=_TEMPLATE_PLAYER, name="Thessaly Vane", race="Elf", character_class="Ranger",
        abilities=Abilities(strength=12, dexterity=17, constitution=13, intelligence=10, wisdom=15, charisma=8),
        skills={Skill.SURVIVAL: P, Skill.NATURE: P, Skill.PERCEPTION: P, Skill.STEALTH: P},
        saves=[Ability.STR, Ability.DEX],
        armor_class=14, max_hp=11, hit_die=10, speed=35,
        weapons=[WEAPONS["longbow"], WEAPONS["shortsword"]],
        inventory=["leather armor", "20 arrows", "backpack", "rope (50 ft)", "hunting trap"],
        notes="Favored Enemy: goblinoids. Natural Explorer: forest.",
    ),
    # Half-orc +2 STR, +1 CON. Unarmored Defense: 10 + DEX + CON.
    "barbarian": Character(
        player_id=_TEMPLATE_PLAYER, name="Grask Bonecrusher", race="Half-Orc", character_class="Barbarian",
        abilities=Abilities(strength=17, dexterity=13, constitution=16, intelligence=8, wisdom=12, charisma=10),
        skills={Skill.ATHLETICS: P, Skill.INTIMIDATION: P, Skill.SURVIVAL: P},
        saves=[Ability.STR, Ability.CON],
        armor_class=14, max_hp=15, hit_die=12, speed=30,
        weapons=[WEAPONS["greataxe"], WEAPONS["javelin"]],
        inventory=["backpack", "4 javelins", "bedroll", "rations (10 days)"],
        notes="Rage 2/long rest: +2 melee damage, resistance to bludgeoning/piercing/slashing. Relentless Endurance.",
    ),
}


def pregen(key: str, player_id: str, name: Optional[str] = None) -> Character:
    """A fresh copy of a template, owned by `player_id`, optionally renamed."""
    try:
        template = PREGENS[key.lower().strip()]
    except KeyError:
        raise KeyError(f"no pregen called {key!r}; choose one of {', '.join(PREGENS)}") from None
    changes = {"player_id": player_id}
    if name:
        changes["name"] = name.strip()
    return template.model_copy(update=changes, deep=True)


def describe_pregens() -> List[str]:
    """One line each, for `/join` to show."""
    return [
        f"{key}: {c.name}, {c.race} {c.character_class} — AC {c.armor_class}, {c.max_hp} HP, "
        f"{', '.join(w.name.lower() for w in c.weapons)}"
        for key, c in PREGENS.items()
    ]
