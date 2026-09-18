"""Combat: initiative, turns, attacks, damage, conditions, death, rests.

Every function here is pure — it takes frozen models and returns new ones
together with a list of `Event`s, one per thing that happened, each with a
`text` the Dungeon Master narrates from. Nothing is mutated; a caller that
ignores the return value has changed nothing.

Illegal actions raise `RulesError` with a sentence a player can read ("It is
goblin-1's turn, not Kara's"). The tool layer (PR-17) turns those into
replies; the engine never guesses what the caller meant.

Rules applied, per SRD 5.1 — each pinned by a test in
`tests/test_engine_combat.py`:

- Initiative is d20 + DEX modifier, highest first; ties by DEX, then name.
- Temporary hit points absorb damage first and do not stack (higher wins).
- Resistance halves damage (rounded down), immunity zeroes it, vulnerability
  doubles it. Matching is by damage type; the "from nonmagical attacks"
  qualifier on many monster resistances is not modelled — every attack here
  is treated as nonmagical.
- A monster at 0 HP is dead. A character at 0 HP falls unconscious and
  starts death saves, unless the damage left over after reaching 0 equals or
  exceeds their maximum HP (instant death).
- Death saves: 10 or more succeeds, a 20 restores 1 HP, a 1 counts as two
  failures; three successes stabilise, three failures kill. Damage while at
  0 HP is a failed save (two on a critical hit).
- Conditions: an incapacitated creature cannot act; blinded, poisoned,
  prone, restrained and frightened attackers have disadvantage; attacks
  against a blinded, restrained, paralyzed, stunned, unconscious or
  petrified target have advantage; a melee hit on a paralyzed or unconscious
  target is a critical; a prone target gives melee advantage and ranged
  disadvantage. Restrained gives disadvantage on DEX saves; paralyzed,
  stunned, unconscious and petrified fail STR and DEX saves outright.
- A short rest spends hit dice for d(hit die) + CON each; a long rest
  restores all HP, half the hit dice (at least one), and clears temporary HP.
"""

import random
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

from src.engine.character import Ability, Character, Condition, ability_modifier
from src.engine.checks import RollMode, resolve_mode, roll_d20
from src.engine.combatant import Attack, Combatant, Damage
from src.utils.dice import DiceRoller, RandomSource

Creature = Union[Character, Combatant]


class RulesError(ValueError):
    """The rules do not allow that right now. The message is for the player."""


class Event(BaseModel):
    """One thing that happened, as a line the DM can narrate from."""

    model_config = ConfigDict(frozen=True)

    kind: str
    text: str
    actor: Optional[str] = None
    target: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)


Events = List[Event]


# --- condition effects ---------------------------------------------------------

INCAPACITATING = {
    Condition.INCAPACITATED, Condition.PARALYZED, Condition.PETRIFIED,
    Condition.STUNNED, Condition.UNCONSCIOUS,
}
ATTACKER_DISADVANTAGE = {
    Condition.BLINDED, Condition.POISONED, Condition.PRONE, Condition.RESTRAINED, Condition.FRIGHTENED,
}
TARGET_GRANTS_ADVANTAGE = {
    Condition.BLINDED, Condition.RESTRAINED, Condition.PARALYZED, Condition.STUNNED,
    Condition.UNCONSCIOUS, Condition.PETRIFIED,
}
TARGET_MELEE_AUTO_CRIT = {Condition.PARALYZED, Condition.UNCONSCIOUS}
CHECK_DISADVANTAGE = {Condition.POISONED, Condition.FRIGHTENED}
AUTO_FAIL_STR_DEX_SAVES = {
    Condition.PARALYZED, Condition.STUNNED, Condition.UNCONSCIOUS, Condition.PETRIFIED,
}


def _label(creature: Creature) -> str:
    """How a creature is named in events: `goblin-2`, not `Goblin` twice over."""
    return getattr(creature, "id", None) or creature.name


def _conditions(creature: Creature) -> set:
    return set(creature.conditions)


def can_act(creature: Creature) -> bool:
    if getattr(creature, "dead", False) or creature.current_hp <= 0:
        return False
    return not (_conditions(creature) & INCAPACITATING)


def why_cannot_act(creature: Creature) -> str:
    if getattr(creature, "dead", False):
        return f"{_label(creature)} is dead."
    if creature.current_hp <= 0:
        return f"{_label(creature)} is unconscious at 0 HP."
    blocking = sorted(c.value for c in _conditions(creature) & INCAPACITATING)
    return f"{_label(creature)} is {', '.join(blocking)} and cannot act."


def attack_mode(attacker: Creature, target: Creature, ranged: bool) -> RollMode:
    """Advantage and disadvantage from the two creatures' conditions."""
    modes = []
    attacking = _conditions(attacker)
    targeted = _conditions(target)
    if attacking & ATTACKER_DISADVANTAGE:
        modes.append(RollMode.DISADVANTAGE)
    if Condition.INVISIBLE in attacking:
        modes.append(RollMode.ADVANTAGE)
    if targeted & TARGET_GRANTS_ADVANTAGE:
        modes.append(RollMode.ADVANTAGE)
    if Condition.INVISIBLE in targeted:
        modes.append(RollMode.DISADVANTAGE)
    if Condition.PRONE in targeted:
        modes.append(RollMode.DISADVANTAGE if ranged else RollMode.ADVANTAGE)
    return resolve_mode(*modes)


def check_mode(creature: Creature) -> RollMode:
    return RollMode.DISADVANTAGE if _conditions(creature) & CHECK_DISADVANTAGE else RollMode.NORMAL


def save_mode(creature: Creature, ability: Ability) -> Optional[RollMode]:
    """The roll mode for a save, or None when the save fails automatically."""
    ability = Ability(ability)
    conditions = _conditions(creature)
    if ability in (Ability.STR, Ability.DEX) and conditions & AUTO_FAIL_STR_DEX_SAVES:
        return None
    if ability == Ability.DEX and Condition.RESTRAINED in conditions:
        return RollMode.DISADVANTAGE
    return RollMode.NORMAL


# --- hit points -------------------------------------------------------------------

def _is_character(creature: Creature) -> bool:
    return isinstance(creature, Character) or getattr(creature, "kind", "") == "character"


def _tracks_death_saves(creature: Creature) -> bool:
    return isinstance(creature, Combatant) and creature.kind == "character"


def damage_multiplier(creature: Creature, damage_type: str) -> float:
    """1.0, 0.5 (resistant), 0 (immune) or 2.0 (vulnerable) for this type."""
    kind = (damage_type or "").lower().strip()
    if not kind:
        return 1.0

    def listed(field: str) -> bool:
        return any(kind in str(entry).lower() for entry in getattr(creature, field, []) or [])

    if listed("damage_immunities"):
        return 0.0
    if listed("damage_resistances"):
        return 0.5
    if listed("damage_vulnerabilities"):
        return 2.0
    return 1.0


def apply_damage(
    creature: Creature,
    amount: int,
    damage_type: str = "",
    critical: bool = False,
) -> Tuple[Creature, Events]:
    """Damage a creature, honouring temp HP, resistances, dying and death."""
    if getattr(creature, "dead", False):
        raise RulesError(f"{_label(creature)} is already dead.")
    if amount < 0:
        raise RulesError("damage cannot be negative")

    multiplier = damage_multiplier(creature, damage_type)
    dealt = int(amount * multiplier)
    absorbed = min(creature.temp_hp, dealt)
    remaining = dealt - absorbed
    was_down = creature.current_hp == 0
    new_hp = max(0, creature.current_hp - remaining)
    overflow = remaining - creature.current_hp  # damage past 0

    notes = []
    if multiplier == 0:
        notes.append("immune")
    elif multiplier == 0.5:
        notes.append("resisted, halved")
    elif multiplier == 2:
        notes.append("vulnerable, doubled")
    if absorbed:
        notes.append(f"{absorbed} absorbed by temporary HP")
    note = f" ({'; '.join(notes)})" if notes else ""
    label = f" {damage_type}" if damage_type else ""

    events: Events = [
        Event(
            kind="damage",
            target=_label(creature),
            text=f"{_label(creature)} takes {dealt}{label} damage{note} — {new_hp}/{creature.max_hp} HP.",
            data={"amount": dealt, "type": damage_type, "hp": new_hp, "absorbed": absorbed},
        )
    ]
    updated = creature.model_copy(update={"current_hp": new_hp, "temp_hp": creature.temp_hp - absorbed})

    if dealt == 0:
        return updated, events

    if was_down and remaining > 0:
        # Damage while at 0 HP is a failed death save; a crit is two.
        if _tracks_death_saves(creature):
            return _fail_death_saves(updated, 2 if critical else 1, events)
        return updated, events

    if new_hp == 0 and creature.current_hp > 0:
        if not _is_character(creature) or overflow >= creature.max_hp:
            updated = updated.model_copy(update={"dead": True})
            cause = "instantly" if _is_character(creature) else ""
            events.append(Event(kind="died", target=_label(creature), text=f"{_label(creature)} dies{' ' + cause if cause else ''}."))
        else:
            conditions = list(creature.conditions)
            if Condition.UNCONSCIOUS not in conditions:
                conditions.append(Condition.UNCONSCIOUS)
            changes: Dict[str, Any] = {"conditions": conditions}
            if _tracks_death_saves(creature):
                changes.update(death_successes=0, death_failures=0, stable=False)
            updated = updated.model_copy(update=changes)
            events.append(Event(kind="dropped", target=_label(creature), text=f"{_label(creature)} drops to 0 HP and falls unconscious."))

    return updated, events


def _fail_death_saves(combatant: Combatant, count: int, events: Events) -> Tuple[Combatant, Events]:
    failures = combatant.death_failures + count
    updated = combatant.model_copy(update={"death_failures": failures, "stable": False})
    events.append(
        Event(
            kind="death_save",
            target=_label(combatant),
            text=f"{_label(combatant)} suffers {count} death save failure{'s' if count > 1 else ''} ({updated.death_successes} successes, {failures} failures).",
            data={"successes": updated.death_successes, "failures": failures},
        )
    )
    if failures >= 3:
        updated = updated.model_copy(update={"dead": True})
        events.append(Event(kind="died", target=_label(combatant), text=f"{_label(combatant)} dies."))
    return updated, events


def heal(creature: Creature, amount: int) -> Tuple[Creature, Events]:
    if getattr(creature, "dead", False):
        raise RulesError(f"{_label(creature)} is dead; healing cannot help.")
    if amount < 0:
        raise RulesError("healing cannot be negative")
    new_hp = min(creature.max_hp, creature.current_hp + amount)
    changes: Dict[str, Any] = {"current_hp": new_hp}
    text = f"{_label(creature)} regains {new_hp - creature.current_hp} HP — {new_hp}/{creature.max_hp}."
    if creature.current_hp == 0 and new_hp > 0:
        changes["conditions"] = [c for c in creature.conditions if c != Condition.UNCONSCIOUS]
        if _tracks_death_saves(creature):
            changes.update(death_successes=0, death_failures=0, stable=False)
        text += f" {_label(creature)} regains consciousness."
    return creature.model_copy(update=changes), [Event(kind="healed", target=_label(creature), text=text, data={"hp": new_hp})]


def grant_temp_hp(creature: Creature, amount: int) -> Tuple[Creature, Events]:
    """Temporary HP do not stack: the higher value stands."""
    new_temp = max(creature.temp_hp, amount)
    text = (
        f"{_label(creature)} gains {amount} temporary HP."
        if new_temp > creature.temp_hp
        else f"{_label(creature)} already has {creature.temp_hp} temporary HP; {amount} would not be more."
    )
    return creature.model_copy(update={"temp_hp": new_temp}), [Event(kind="temp_hp", target=_label(creature), text=text, data={"temp_hp": new_temp})]


def death_save(combatant: Combatant, rng: Optional[RandomSource] = None) -> Tuple[Combatant, Events]:
    if combatant.dead:
        raise RulesError(f"{_label(combatant)} is dead.")
    if combatant.current_hp > 0:
        raise RulesError(f"{_label(combatant)} is not dying.")
    if combatant.stable:
        raise RulesError(f"{_label(combatant)} is stable and does not need to roll.")

    roll, _ = roll_d20(RollMode.NORMAL, rng)
    if roll == 20:
        updated = combatant.model_copy(update={
            "current_hp": 1, "death_successes": 0, "death_failures": 0, "stable": False,
            "conditions": [c for c in combatant.conditions if c != Condition.UNCONSCIOUS],
        })
        return updated, [Event(kind="death_save", target=_label(combatant), text=f"{_label(combatant)} rolls a natural 20 on a death save and regains consciousness with 1 HP!", data={"roll": 20})]
    if roll == 1:
        return _fail_death_saves(combatant, 2, [Event(kind="death_save", target=_label(combatant), text=f"{_label(combatant)} rolls a 1 on a death save — two failures.", data={"roll": 1})])
    if roll >= 10:
        successes = combatant.death_successes + 1
        updated = combatant.model_copy(update={"death_successes": successes})
        events = [Event(kind="death_save", target=_label(combatant), text=f"{_label(combatant)} succeeds a death save ({roll}) — {successes} successes, {combatant.death_failures} failures.", data={"roll": roll, "successes": successes, "failures": combatant.death_failures})]
        if successes >= 3:
            updated = updated.model_copy(update={"stable": True, "death_successes": 0, "death_failures": 0})
            events.append(Event(kind="stabilized", target=_label(combatant), text=f"{_label(combatant)} is stable."))
        return updated, events
    return _fail_death_saves(combatant, 1, [Event(kind="death_save", target=_label(combatant), text=f"{_label(combatant)} fails a death save ({roll}).", data={"roll": roll})])


def stabilize(combatant: Combatant) -> Tuple[Combatant, Events]:
    """A successful Medicine check or a Spare the Dying: stable at 0 HP."""
    if combatant.dead or combatant.current_hp > 0:
        raise RulesError(f"{_label(combatant)} does not need stabilising.")
    updated = combatant.model_copy(update={"stable": True, "death_successes": 0, "death_failures": 0})
    return updated, [Event(kind="stabilized", target=_label(combatant), text=f"{_label(combatant)} is stabilised.")]


# --- conditions ---------------------------------------------------------------------

def add_condition(creature: Creature, condition: Condition) -> Tuple[Creature, Events]:
    condition = Condition(condition)
    if condition in creature.conditions:
        return creature, []
    immune = [c for c in getattr(creature, "condition_immunities", []) or [] if condition.value in str(c).lower()]
    if immune:
        return creature, [Event(kind="condition_immune", target=_label(creature), text=f"{_label(creature)} is immune to being {condition.value}.")]
    updated = creature.model_copy(update={"conditions": [*creature.conditions, condition]})
    return updated, [Event(kind="condition_added", target=_label(creature), text=f"{_label(creature)} is now {condition.value}.", data={"condition": condition.value})]


def remove_condition(creature: Creature, condition: Condition) -> Tuple[Creature, Events]:
    condition = Condition(condition)
    if condition not in creature.conditions:
        return creature, []
    updated = creature.model_copy(update={"conditions": [c for c in creature.conditions if c != condition]})
    return updated, [Event(kind="condition_removed", target=_label(creature), text=f"{_label(creature)} is no longer {condition.value}.", data={"condition": condition.value})]


# --- the encounter ----------------------------------------------------------------

class Encounter(BaseModel):
    """Who is fighting, in what order, and whose turn it is. Frozen."""

    model_config = ConfigDict(frozen=True)

    combatants: Dict[str, Combatant]
    order: List[str]  # ids, highest initiative first
    initiative: Dict[str, int]
    round: int = 1
    turn_index: int = 0
    active: bool = True
    outcome: Optional[str] = None  # "victory" | "defeat" | "fled"

    @property
    def current(self) -> str:
        return self.order[self.turn_index]

    def get(self, combatant_id: str) -> Combatant:
        key = (combatant_id or "").strip()
        for candidate in (key, key.lower()):
            if candidate in self.combatants:
                return self.combatants[candidate]
        for combatant in self.combatants.values():
            if combatant.name.lower() == key.lower():
                return combatant
        raise RulesError(f"No one called {combatant_id!r} is in this fight. Present: {', '.join(self.order)}.")

    @property
    def characters(self) -> List[Combatant]:
        return [c for c in self.combatants.values() if c.kind == "character"]

    @property
    def monsters(self) -> List[Combatant]:
        return [c for c in self.combatants.values() if c.kind == "monster"]

    def sheet(self) -> str:
        """The scene sheet the DM reads: order, whose turn, everyone's state."""
        if not self.active:
            return f"The fight is over ({self.outcome})."
        lines = [f"Round {self.round} — {self.current}'s turn."]
        lines.append("Order: " + ", ".join(f"{i} ({self.initiative[i]})" for i in self.order))
        for combatant_id in self.order:
            c = self.combatants[combatant_id]
            marker = "✝ " if c.dead else ""
            lines.append(f"  {marker}{c.sheet_line()}")
        return "\n".join(lines)


def _with(encounter: Encounter, *combatants: Combatant, **changes: Any) -> Encounter:
    updated = dict(encounter.combatants)
    for combatant in combatants:
        updated[combatant.id] = combatant
    return encounter.model_copy(update={"combatants": updated, **changes})


def start_encounter(
    party: List[Character],
    monsters: List[Combatant],
    rng: Optional[RandomSource] = None,
) -> Tuple[Encounter, Events]:
    """Roll initiative for everyone and open round 1."""
    source = rng if rng is not None else random
    combatants: Dict[str, Combatant] = {}
    for character in party:
        combatant = Combatant.from_character(character)
        if combatant.id in combatants:
            raise RulesError(f"Two combatants share the name {combatant.id!r}.")
        combatants[combatant.id] = combatant
    for monster in monsters:
        if monster.id in combatants:
            raise RulesError(f"Two combatants share the id {monster.id!r}.")
        combatants[monster.id] = monster
    if not combatants:
        raise RulesError("Nobody is here to fight.")

    events: Events = [Event(kind="encounter_started", text="Roll for initiative!")]
    initiative: Dict[str, int] = {}
    for combatant_id, combatant in combatants.items():
        roll = source.randint(1, 20)
        initiative[combatant_id] = roll + combatant.initiative_modifier
        events.append(
            Event(
                kind="initiative",
                actor=_label(combatant),
                text=f"{_label(combatant)} rolls initiative: d20 {roll} {'+' if combatant.initiative_modifier >= 0 else '−'} {abs(combatant.initiative_modifier)} = {initiative[combatant_id]}.",
                data={"initiative": initiative[combatant_id]},
            )
        )

    order = sorted(
        combatants,
        key=lambda i: (-initiative[i], -combatants[i].initiative_modifier, combatants[i].name),
    )
    encounter = Encounter(combatants=combatants, order=order, initiative=initiative)
    events.append(Event(kind="round_started", text="Round 1.", data={"round": 1}))
    events.append(Event(kind="turn_started", actor=_label(combatants[order[0]]), text=f"{_label(combatants[order[0]])} acts first."))
    return encounter, events


def move_first(encounter: Encounter, combatant_id: str) -> Tuple[Encounter, Events]:
    """Put a combatant at the top of the order: they opened the fight.

    A player who declares an attack on an unsuspecting creature has, in
    effect, surprised it — their blow lands before initiative plays out.
    """
    actor = encounter.get(combatant_id)
    if encounter.order[0] == actor.id:
        return encounter, []
    order = [actor.id] + [i for i in encounter.order if i != actor.id]
    moved = encounter.model_copy(update={"order": order, "turn_index": 0})
    return moved, [Event(kind="surprise", actor=actor.id, text=f"{actor.id} strikes before anyone can react.")]


def _check_end(encounter: Encounter) -> Tuple[Encounter, Events]:
    if not encounter.active:
        return encounter, []
    monsters_left = [m for m in encounter.monsters if not m.dead]
    characters_up = [c for c in encounter.characters if not c.dead and c.current_hp > 0]
    if encounter.monsters and not monsters_left:
        ended = encounter.model_copy(update={"active": False, "outcome": "victory"})
        return ended, [Event(kind="encounter_ended", text="The last enemy falls. Victory!", data={"outcome": "victory"})]
    if encounter.characters and not characters_up:
        ended = encounter.model_copy(update={"active": False, "outcome": "defeat"})
        return ended, [Event(kind="encounter_ended", text="The whole party is down.", data={"outcome": "defeat"})]
    return encounter, []


def _roll_component(component: Damage, critical: bool, rng: Optional[RandomSource]) -> Tuple[int, List[int], str]:
    """(total, dice rolled, notation) for one damage component."""
    if component.dice is None:
        return max(0, component.bonus), [], str(component.bonus)
    notation = component.dice
    if critical:
        notation = "+".join(f"{q * 2}d{s}" for q, s in DiceRoller.parse_dice_string(component.dice))
    rolled = DiceRoller.roll_multiple(notation, rng)
    dice = [r for group in rolled for r in group.results]
    return max(0, sum(dice) + component.bonus), dice, notation


def attack(
    encounter: Encounter,
    attacker_id: str,
    target_id: str,
    attack_name: Optional[str] = None,
    mode: RollMode = RollMode.NORMAL,
    rng: Optional[RandomSource] = None,
    enforce_turn: bool = True,
) -> Tuple[Encounter, Events]:
    """One attack on the attacker's turn, damage applied on a hit."""
    if not encounter.active:
        raise RulesError("The fight is over.")
    attacker = encounter.get(attacker_id)
    target = encounter.get(target_id)
    if enforce_turn and encounter.current != attacker.id:
        raise RulesError(f"It is {_label(encounter.get(encounter.current))}'s turn, not {_label(attacker)}'s.")
    if not can_act(attacker):
        raise RulesError(why_cannot_act(attacker))
    if target.dead:
        raise RulesError(f"{_label(target)} is already dead.")
    if attacker.id == target.id:
        raise RulesError(f"{_label(attacker)} cannot attack themselves.")
    if not attacker.attacks:
        raise RulesError(f"{_label(attacker)} has no attack to make.")
    chosen: Attack = attacker.attack(attack_name) if attack_name else attacker.attacks[0]

    final_mode = resolve_mode(RollMode(mode), attack_mode(attacker, target, chosen.ranged))
    kept, rolls = roll_d20(final_mode, rng)
    total = kept + chosen.attack_bonus
    auto_crit = not chosen.ranged and bool(_conditions(target) & TARGET_MELEE_AUTO_CRIT)

    if kept == 1:
        hit, critical = False, False
    elif kept == 20:
        hit, critical = True, True
    else:
        hit = total >= target.armor_class
        critical = hit and auto_crit

    dice_text = f"d20 {list(rolls)}→{kept}" if len(rolls) > 1 else f"d20 {kept}"
    mode_text = f" ({final_mode.value})" if final_mode != RollMode.NORMAL else ""
    outcome = "critical hit" if critical else ("hit" if hit else "miss")
    events: Events = [
        Event(
            kind="attack",
            actor=_label(attacker),
            target=_label(target),
            text=f"{_label(attacker)} attacks {_label(target)} with {chosen.name}: {dice_text} + {chosen.attack_bonus} = {total}{mode_text} vs AC {target.armor_class} — {outcome}.",
            data={"roll": kept, "rolls": list(rolls), "total": total, "hit": hit, "critical": critical, "mode": final_mode.value},
        )
    ]

    updated_target = target
    if hit:
        for component in chosen.damage:
            amount, dice, notation = _roll_component(component, critical, rng)
            events.append(
                Event(
                    kind="damage_rolled",
                    actor=_label(attacker),
                    target=_label(target),
                    text=f"{chosen.name} damage: {notation} {dice} {'+' if component.bonus >= 0 else '−'} {abs(component.bonus)} = {amount} {component.damage_type}{' (critical: dice doubled)' if critical and dice else ''}.",
                    data={"amount": amount, "type": component.damage_type, "dice": dice},
                )
            )
            updated_target, more = apply_damage(updated_target, amount, component.damage_type, critical=critical)
            events.extend(more)
            if updated_target.dead:
                break

    result = _with(encounter, updated_target)
    result, ended = _check_end(result)
    return result, events + ended


def damage_combatant(
    encounter: Encounter, target_id: str, amount: int, damage_type: str = "", critical: bool = False
) -> Tuple[Encounter, Events]:
    """Damage from something other than an attack roll: a trap, a spell, a fall."""
    if not encounter.active:
        raise RulesError("The fight is over.")
    target, events = apply_damage(encounter.get(target_id), amount, damage_type, critical)
    result, ended = _check_end(_with(encounter, target))
    return result, events + ended


def heal_combatant(encounter: Encounter, target_id: str, amount: int) -> Tuple[Encounter, Events]:
    target, events = heal(encounter.get(target_id), amount)
    return _with(encounter, target), events


def end_turn(encounter: Encounter, rng: Optional[RandomSource] = None) -> Tuple[Encounter, Events]:
    """Advance to the next combatant who can act.

    Dead combatants are skipped. A character at 0 HP gets their death save
    rolled as their turn begins and is then skipped, since they cannot act.
    """
    if not encounter.active:
        raise RulesError("The fight is over.")
    events: Events = [Event(kind="turn_ended", actor=_label(encounter.get(encounter.current)), text=f"{_label(encounter.get(encounter.current))}'s turn ends.")]
    current = encounter

    for _ in range(len(encounter.order) + 1):
        index = current.turn_index + 1
        round_number = current.round
        if index >= len(current.order):
            index = 0
            round_number += 1
            events.append(Event(kind="round_started", text=f"Round {round_number}.", data={"round": round_number}))
        current = current.model_copy(update={"turn_index": index, "round": round_number})

        combatant = current.get(current.current)
        if combatant.dead:
            continue
        if combatant.kind == "character" and combatant.current_hp == 0:
            if not combatant.stable:
                combatant, saves = death_save(combatant, rng)
                events.extend(saves)
                current = _with(current, combatant)
                current, ended = _check_end(current)
                events.extend(ended)
                if not current.active:
                    return current, events
            continue
        if not can_act(combatant):
            events.append(Event(kind="turn_skipped", actor=_label(combatant), text=why_cannot_act(combatant)))
            continue
        events.append(Event(kind="turn_started", actor=_label(combatant), text=f"{_label(combatant)}'s turn."))
        return current, events

    current, ended = _check_end(current)
    if current.active:
        current = current.model_copy(update={"active": False, "outcome": "stalemate"})
        ended = [Event(kind="encounter_ended", text="No one left can act.", data={"outcome": "stalemate"})]
    return current, events + ended


def end_encounter(encounter: Encounter, outcome: str = "fled") -> Tuple[Encounter, Events]:
    """Close a fight by fiat: the party ran, the monsters surrendered."""
    if not encounter.active:
        return encounter, []
    return encounter.model_copy(update={"active": False, "outcome": outcome}), [
        Event(kind="encounter_ended", text=f"The fight ends ({outcome}).", data={"outcome": outcome})
    ]


def sync_party(party: Dict[str, Character], encounter: Encounter) -> Dict[str, Character]:
    """Write the fight's HP, temp HP, conditions and deaths back to the sheets."""
    synced = dict(party)
    for combatant in encounter.characters:
        for key, character in party.items():
            if character.name == combatant.name:
                synced[key] = character.model_copy(update={
                    "current_hp": combatant.current_hp,
                    "temp_hp": combatant.temp_hp,
                    "conditions": list(combatant.conditions),
                    "dead": combatant.dead,
                })
    return synced


# --- rests ---------------------------------------------------------------------------

def short_rest(character: Character, hit_dice: int = 1, rng: Optional[RandomSource] = None) -> Tuple[Character, Events]:
    """Spend hit dice: each heals d(hit die) + CON modifier, never less than 0."""
    if character.dead:
        raise RulesError(f"{character.name} is dead.")
    if hit_dice < 0:
        raise RulesError("cannot spend a negative number of hit dice")
    spend = min(hit_dice, character.hit_dice_remaining)
    if hit_dice and not spend:
        raise RulesError(f"{character.name} has no hit dice left to spend.")
    source = rng if rng is not None else random
    con = ability_modifier(character.abilities.constitution)
    healed = 0
    rolls = []
    for _ in range(spend):
        roll = source.randint(1, character.hit_die)
        rolls.append(roll)
        healed += max(0, roll + con)
    new_hp = min(character.max_hp, character.current_hp + healed)
    updated = character.model_copy(update={"current_hp": new_hp, "hit_dice_remaining": character.hit_dice_remaining - spend})
    text = f"{character.name} takes a short rest"
    if spend:
        text += f", spending {spend} hit {'die' if spend == 1 else 'dice'} (d{character.hit_die} {rolls} {'+' if con >= 0 else '−'} {abs(con)} each) to regain {new_hp - character.current_hp} HP — {new_hp}/{character.max_hp}, {updated.hit_dice_remaining} hit dice left"
    return updated, [Event(kind="rest", actor=character.name, text=text + ".", data={"hp": new_hp, "hit_dice_remaining": updated.hit_dice_remaining})]


def long_rest(character: Character) -> Tuple[Character, Events]:
    """Full HP, half the hit dice back (at least one), temporary HP gone."""
    if character.dead:
        raise RulesError(f"{character.name} is dead.")
    regained = max(1, character.level // 2)
    updated = character.model_copy(update={
        "current_hp": character.max_hp,
        "temp_hp": 0,
        "hit_dice_remaining": min(character.level, character.hit_dice_remaining + regained),
        "conditions": [c for c in character.conditions if c != Condition.UNCONSCIOUS],
    })
    return updated, [Event(kind="rest", actor=character.name, text=f"{character.name} takes a long rest and wakes at full health: {updated.max_hp}/{updated.max_hp} HP, {updated.hit_dice_remaining} hit dice.", data={"hp": updated.max_hp})]
