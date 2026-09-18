"""The Dungeon Master's tools: thin, validated wrappers over the engine.

Each tool is a plain function ``tool(state, *, args..., rng=None) ->
ToolResult``. It reads the engine models out of ``GameState`` through the
typed accessors, calls the engine, and returns

- ``text`` — a few lines the model reads and narrates from, never a number
  it has to compute;
- ``update`` — the state delta to write (``party`` / ``encounter`` /
  ``pending`` as JSON dicts), empty for a read-only tool;
- ``events`` — the engine events, for the log.

Arguments are declared as pydantic models (``AttackArgs`` and so on) so the
same schema validates a call *and* describes the tool to the model in PR-18.
Nothing here calls a language model. ``lookup_rules`` embeds the question
through the daemon and returns the retrieved passages — retrieval, not
generation — and takes an injectable retriever so tests stay offline.

Errors never escape: ``registry.run_tool`` validates the arguments and turns
``RulesError``, ``UnknownEntry``, ``KeyError`` and ``ValidationError`` into a
sentence the model can act on. The functions here raise; the dispatcher
speaks.
"""

import re
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.engine import combat
from src.engine.character import ABILITY_NAMES, Ability, Character, Condition, Skill
from src.engine.checks import CheckResult, PendingCheck, RollMode
from src.engine.combat import Encounter, Event, RulesError
from src.graph.game_state import (
    get_encounter,
    get_party,
    get_pending,
    put_encounter,
    put_party,
    put_pending,
)
from src.srd import monster
from src.srd.bestiary import summon
from src.utils.dice import RandomSource

State = Mapping[str, Any]


@dataclass(frozen=True)
class ToolResult:
    text: str
    update: Dict[str, Any] = field(default_factory=dict)
    events: List[Event] = field(default_factory=list)
    ok: bool = True


def _lines(events: List[Event]) -> str:
    return "\n".join(e.text for e in events)


def _find_character(party: Dict[str, Character], name: str) -> Character:
    key = (name or "").strip().lower()
    for character in party.values():
        if character.name.lower() == key:
            return character
    for stored, character in party.items():
        if stored.lower() == key:
            return character
    present = ", ".join(c.name for c in party.values()) or "nobody"
    raise RulesError(f"No party member called {name!r}. In the party: {present}.")


def _store_party(party: Dict[str, Character], character: Character) -> Dict[str, Character]:
    updated = dict(party)
    for stored, existing in party.items():
        if existing.name == character.name:
            updated[stored] = character
            return updated
    updated[character.name] = character
    return updated


def _after_encounter_action(
    state: State, party: Dict[str, Character], encounter: Encounter, events: List[Event]
) -> ToolResult:
    """Write an encounter back; when it has ended, sync the sheets and clear it."""
    text = _lines(events)
    if encounter.active:
        return ToolResult(text=text, update={"encounter": put_encounter(encounter)}, events=events)
    synced = combat.sync_party(party, encounter)
    return ToolResult(
        text=text + f"\nThe encounter is over ({encounter.outcome}).",
        update={"encounter": None, "party": put_party(synced)},
        events=events,
    )


# --- read-only ----------------------------------------------------------------------

class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


def get_scene(state: State, rng: Optional[RandomSource] = None) -> ToolResult:
    """The party, the fight if there is one, and any roll being waited on."""
    party = get_party(state)
    encounter = get_encounter(state)
    pending = get_pending(state)
    world = state.get("game_state") or {}

    lines: List[str] = []
    if isinstance(world, dict) and world.get("location"):
        lines.append(f"Location: {world['location']}.")
    if party:
        lines.append("Party:")
        lines.extend(f"  {c.sheet_line()}" for c in party.values())
    else:
        lines.append("No one has joined the party yet.")
    if encounter:
        lines.append(encounter.sheet())
    else:
        lines.append("Not in combat.")
    if pending:
        lines.append(f"Waiting on {pending.player} for a {pending.label} (DC {pending.dc}).")
    return ToolResult(text="\n".join(lines))


class LookupMonsterArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(description="The creature's name as in the SRD, e.g. 'goblin', 'adult red dragon'.")


def lookup_monster(state: State, name: str, rng: Optional[RandomSource] = None) -> ToolResult:
    """A stat block summary from the SRD. Read-only."""
    creature = summon(name)
    lines = [
        f"{creature.name} — {creature.size} {creature.creature_type}, {creature.alignment}. CR {creature.challenge_rating:g} ({creature.xp} XP).",
        f"AC {creature.armor_class}, {creature.max_hp} HP, speed {', '.join(f'{k} {v}' for k, v in creature.speed.items())}.",
    ]
    if creature.attacks:
        lines.append("Attacks: " + "; ".join(a.describe() for a in creature.attacks) + ".")
    if creature.multiattack:
        lines.append(f"Multiattack: {creature.multiattack}")
    for trait in creature.traits[:4]:
        lines.append(f"Trait: {trait}")
    for label, values in (
        ("Resistances", creature.damage_resistances),
        ("Immunities", creature.damage_immunities),
        ("Vulnerabilities", creature.damage_vulnerabilities),
    ):
        if values:
            lines.append(f"{label}: {', '.join(values)}.")
    return ToolResult(text="\n".join(lines))


class LookupRulesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(description="The rules question, in plain words.")


Retriever = Callable[[str], List[Any]]  # question -> Documents


def _default_retriever() -> Retriever:
    """Similarity search over the SRD index. Embeds the question via the daemon;
    no generation, no rewrite (the rewriter would be a model call)."""
    from src.agents.researcher import RETRIEVAL_K
    from src.data.vectorstore import load_vectorstore

    store = load_vectorstore()
    return lambda question: [doc for doc, _ in store.similarity_search_with_relevance_scores(question, k=RETRIEVAL_K)]


def lookup_rules(
    state: State,
    question: str,
    retriever: Optional[Retriever] = None,
    rng: Optional[RandomSource] = None,
) -> ToolResult:
    """The SRD passages closest to a question, labelled by source."""
    from src.agents.researcher import ResearcherAgent

    docs = (retriever or _default_retriever())(question)
    if not docs:
        return ToolResult(text="The SRD has nothing close to that question.")
    passages = "\n\n".join(
        f"[{ResearcherAgent.citation_for(doc)}]\n{doc.page_content.strip()}" for doc in docs
    )
    return ToolResult(text=f"Passages from the SRD 5.1:\n\n{passages}")


# --- checks the DM asks for --------------------------------------------------------------

ABILITY_HELP = "One of STR, DEX, CON, INT, WIS, CHA."
ATTACK_WORDS = re.compile(r"\b(hit|attack|attacks|strike|strikes|swing|swings|stab|stabs|shoot|shoots)\b", re.IGNORECASE)


class RequestCheckArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    player: str = Field(description="The character who must roll.")
    ability: Ability = Field(description=ABILITY_HELP)
    skill: Optional[Skill] = Field(None, description="A skill such as 'Stealth' or 'Perception', if one applies.")
    dc: int = Field(ge=1, le=40, description="The difficulty class. 10 easy, 15 moderate, 20 hard.")
    reason: str = Field("", description="What the roll is for, in a few words.")


def request_check(
    state: State,
    player: str,
    ability: Ability,
    dc: int,
    skill: Optional[Skill] = None,
    reason: str = "",
    rng: Optional[RandomSource] = None,
) -> ToolResult:
    """Ask a player for an ability check. The DC is recorded, never shown."""
    character = _find_character(get_party(state), player)
    if not combat.can_act(character) and character.current_hp == 0:
        raise RulesError(combat.why_cannot_act(character))
    # Measured on qwen2.5:7b: it reaches for a "Strength check to hit" instead
    # of an attack roll. An attack is never a check; send it to the right tool.
    if get_encounter(state) is not None and ATTACK_WORDS.search(reason or ""):
        raise RulesError(
            f"attacks are not ability checks. Use attack(attacker={character.name!r}, target=<who>) instead."
        )
    pending = PendingCheck(
        player=character.name, kind="check", ability=Ability(ability),
        skill=Skill(skill) if skill else None, dc=dc, mode=combat.check_mode(character), reason=reason,
    )
    return ToolResult(
        text=f"Asked {character.name} for a {pending.label}{' — ' + reason if reason else ''}. Wait for the roll before narrating the outcome.",
        update={"pending": put_pending(pending)},
    )


class RequestSaveArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    player: str = Field(description="The character who must roll.")
    ability: Ability = Field(description=ABILITY_HELP)
    dc: int = Field(ge=1, le=40, description="The difficulty class of the effect.")
    reason: str = Field("", description="What the save is against, in a few words.")


def request_save(
    state: State,
    player: str,
    ability: Ability,
    dc: int,
    reason: str = "",
    rng: Optional[RandomSource] = None,
) -> ToolResult:
    """Ask a player for a saving throw. Auto-failing saves are resolved at once."""
    character = _find_character(get_party(state), player)
    mode = combat.save_mode(character, ability)
    if mode is None:
        return ToolResult(
            text=f"{character.name} automatically fails the {ABILITY_NAMES[Ability(ability)]} saving throw ({', '.join(c.value for c in character.conditions)}).",
        )
    pending = PendingCheck(player=character.name, kind="save", ability=Ability(ability), dc=dc, mode=mode, reason=reason)
    return ToolResult(
        text=f"Asked {character.name} for a {pending.label}{' — ' + reason if reason else ''}. Wait for the roll before narrating the outcome.",
        update={"pending": put_pending(pending)},
    )


class ResolveCheckArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    d20: Optional[int] = Field(None, ge=1, le=20, description="The die the player rolled, or empty to roll for them.")


def resolve_check(state: State, d20: Optional[int] = None, rng: Optional[RandomSource] = None) -> ToolResult:
    """Resolve the pending roll. Called by the player's /roll, not by the DM."""
    pending = get_pending(state)
    if pending is None:
        raise RulesError("Nobody has been asked for a roll.")
    character = _find_character(get_party(state), pending.player)
    if d20 is not None:
        result: CheckResult = pending.resolve(character, rng=_FixedRng(d20))
        result = CheckResult(result.kind, result.label, RollMode.NORMAL, (d20,), d20, result.modifier, d20 + result.modifier, result.dc, d20 + result.modifier >= result.dc)
    else:
        result = pending.resolve(character, rng=rng)
    text = f"{character.name}: {result.describe()}"
    if pending.reason:
        text += f" ({pending.reason})"
    return ToolResult(text=text, update={"pending": None})


class _FixedRng:
    """A die that has already been rolled: the player told us the number."""

    def __init__(self, value: int):
        self.value = value

    def randint(self, low: int, high: int) -> int:
        return self.value


# --- combat ----------------------------------------------------------------------------

class StartEncounterArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    monsters: List[str] = Field(min_length=1, description="SRD monster names, one per creature, e.g. ['goblin', 'goblin', 'wolf'].")


def start_encounter(state: State, monsters: List[str], rng: Optional[RandomSource] = None) -> ToolResult:
    """Roll initiative for the party and the named monsters."""
    if get_encounter(state):
        raise RulesError("A fight is already under way. End it before starting another.")
    party = get_party(state)
    if not party:
        raise RulesError("Nobody is in the party yet.")
    counts: Dict[str, int] = {}
    creatures = []
    for name in monsters:
        entry = monster(name)
        counts[entry["index"]] = counts.get(entry["index"], 0) + 1
        creatures.append(summon(name, combatant_id=f"{entry['index']}-{counts[entry['index']]}", rng=rng))
    encounter, events = combat.start_encounter(list(party.values()), creatures, rng=rng)
    encounter, more = _monsters_act(encounter, rng)  # if a monster won initiative
    events = events + more
    if not encounter.active:
        result = _after_encounter_action(state, party, encounter, events)
        return ToolResult(text=result.text, update={**result.update, "pending": None}, events=events)
    return ToolResult(
        text=_lines(events) + "\n" + encounter.sheet(),
        update={"encounter": put_encounter(encounter), "pending": None},
        events=events,
    )


# A turn is never handed to the model with a monster up. The model was asked
# to call `attack` for monsters and `end_turn` after; measured on qwen2.5:7b it
# narrated the goblin's swing instead of resolving it, then invented the
# outcome. Monster turns are mechanical, so the engine takes them: each monster
# attacks one conscious character (chosen with the game's rng, so a seeded
# session replays), the turn advances, and this repeats until a character is
# up or the fight is over. Multiattack is not modelled: one attack per turn.
MAX_AUTOPILOT_TURNS = 50


def _monsters_act(encounter: Encounter, rng: Optional[RandomSource]) -> tuple:
    events: List[Event] = []
    source = rng if rng is not None else random
    for _ in range(MAX_AUTOPILOT_TURNS):
        if not encounter.active or encounter.get(encounter.current).kind != "monster":
            break
        actor = encounter.get(encounter.current)
        targets = [c for c in encounter.characters if not c.dead and c.current_hp > 0]
        if targets and combat.can_act(actor) and actor.attacks:
            target = targets[source.randint(0, len(targets) - 1)]
            encounter, more = combat.attack(encounter, actor.id, target.id, rng=rng)
            events.extend(more)
        if encounter.active:
            encounter, more = combat.end_turn(encounter, rng=rng)
            events.extend(more)
    return encounter, events


def _require_encounter(state: State) -> Encounter:
    encounter = get_encounter(state)
    if encounter is None:
        raise RulesError("There is no fight going on. Use start_encounter first.")
    return encounter


class AttackArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    attacker: str = Field(description="Who attacks: a character name or a monster id like 'goblin-1'.")
    target: str = Field(description="Who is attacked.")
    attack_name: Optional[str] = Field(None, description="Which weapon or attack, e.g. 'longsword', 'bite'. Defaults to the first.")
    mode: Literal["normal", "advantage", "disadvantage"] = Field("normal", description="Situational advantage or disadvantage, beyond what conditions already give.")


def attack(
    state: State,
    attacker: str,
    target: str,
    attack_name: Optional[str] = None,
    mode: str = "normal",
    rng: Optional[RandomSource] = None,
) -> ToolResult:
    """A player's attack. It ends their turn; the monsters then act until a
    player is up again, and everything that happened is reported.

    With no fight under way, the fight starts here, against the creature
    named as the target. Measured on qwen2.5:7b: told "use start_encounter
    first", the model narrated instead — a refusal that expects a second tool
    call is a refusal a 7B does not recover from.
    """
    encounter = get_encounter(state)
    events: List[Event] = []
    if encounter is None:
        party = get_party(state)
        if not party:
            raise RulesError("Nobody is in the party yet.")
        try:
            creature = summon(target, combatant_id=f"{monster(target)['index']}-1", rng=rng)
        except Exception:
            raise RulesError(
                f"there is no fight going on, and {target!r} is not a creature I know. "
                f"Use start_encounter with the monsters present."
            ) from None
        encounter, events = combat.start_encounter(list(party.values()), [creature], rng=rng)
        # The declared blow opens the fight: the attacker goes first.
        encounter, more = combat.move_first(encounter, attacker)
        events = events + more
        target = creature.id
    encounter, more = combat.attack(encounter, attacker, target, attack_name, RollMode(mode), rng=rng)
    events = events + more
    if encounter.active and encounter.get(attacker).kind == "character":
        encounter, more = combat.end_turn(encounter, rng=rng)
        events = events + more
        encounter, more = _monsters_act(encounter, rng)
        events = events + more
    result = _after_encounter_action(state, get_party(state), encounter, events)
    if encounter.active:
        return ToolResult(text=result.text + f"\nIt is now {encounter.current}'s turn.", update=result.update, events=events)
    return result


class EndTurnArgs(NoArgs):
    pass


def end_turn(state: State, rng: Optional[RandomSource] = None) -> ToolResult:
    """The current player's turn is over without an attack (they hid, dashed,
    talked). The monsters then act until a player is up again."""
    encounter = _require_encounter(state)
    encounter, events = combat.end_turn(encounter, rng=rng)
    encounter, more = _monsters_act(encounter, rng)
    events = events + more
    result = _after_encounter_action(state, get_party(state), encounter, events)
    if encounter.active:
        return ToolResult(text=result.text + f"\nIt is now {encounter.current}'s turn.", update=result.update, events=events)
    return result


class EndEncounterArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    outcome: Literal["fled", "surrendered", "truce", "victory", "defeat"] = Field("fled", description="How the fight ended.")


def end_encounter(state: State, outcome: str = "fled", rng: Optional[RandomSource] = None) -> ToolResult:
    """Stop the fight without a last blow: the party ran, the enemy yielded."""
    encounter = _require_encounter(state)
    encounter, events = combat.end_encounter(encounter, outcome)
    return _after_encounter_action(state, get_party(state), encounter, events)


# --- hit points and conditions, in or out of a fight --------------------------------------

def _apply_to_creature(state: State, target: str, action: Callable) -> ToolResult:
    """Run `action(creature) -> (creature, events)` on whoever `target` is."""
    party = get_party(state)
    encounter = get_encounter(state)
    if encounter is not None:
        creature = encounter.get(target)
        updated, events = action(creature)
        encounter = encounter.model_copy(update={"combatants": {**encounter.combatants, updated.id: updated}})
        encounter, ended = combat._check_end(encounter)
        return _after_encounter_action(state, party, encounter, events + ended)
    character = _find_character(party, target)
    updated, events = action(character)
    return ToolResult(text=_lines(events), update={"party": put_party(_store_party(party, updated))}, events=events)


class ApplyDamageArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: str = Field(description="Who takes the damage.")
    amount: int = Field(ge=0, le=999, description="Hit points of damage, before resistances.")
    damage_type: str = Field("", description="fire, slashing, poison... Empty if untyped.")


def apply_damage(state: State, target: str, amount: int, damage_type: str = "", rng: Optional[RandomSource] = None) -> ToolResult:
    """Damage from something other than an attack roll: a trap, a fall, a spell."""
    return _apply_to_creature(state, target, lambda c: combat.apply_damage(c, amount, damage_type))


class HealArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: str = Field(description="Who is healed.")
    amount: int = Field(ge=0, le=999, description="Hit points restored.")


def heal(state: State, target: str, amount: int, rng: Optional[RandomSource] = None) -> ToolResult:
    """Restore hit points."""
    return _apply_to_creature(state, target, lambda c: combat.heal(c, amount))


class ConditionArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: str = Field(description="Who is affected.")
    condition: Condition = Field(description="One of the SRD conditions: prone, poisoned, restrained, frightened, ...")


def apply_condition(state: State, target: str, condition: Condition, rng: Optional[RandomSource] = None) -> ToolResult:
    """Put a condition on a creature."""
    result = _apply_to_creature(state, target, lambda c: combat.add_condition(c, condition))
    return result if result.text else ToolResult(text=f"{target} is already {Condition(condition).value}.")


def remove_condition(state: State, target: str, condition: Condition, rng: Optional[RandomSource] = None) -> ToolResult:
    """Lift a condition from a creature."""
    result = _apply_to_creature(state, target, lambda c: combat.remove_condition(c, condition))
    return result if result.text else ToolResult(text=f"{target} was not {Condition(condition).value}.")


class RestArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["short", "long"] = Field(description="A short rest (spend hit dice) or a long rest (full recovery).")
    player: Optional[str] = Field(None, description="One character, or empty for the whole party.")
    hit_dice: int = Field(1, ge=0, le=20, description="Hit dice each character spends on a short rest.")


def rest(
    state: State,
    kind: str,
    player: Optional[str] = None,
    hit_dice: int = 1,
    rng: Optional[RandomSource] = None,
) -> ToolResult:
    """A short or long rest for one character or the whole party."""
    if get_encounter(state):
        raise RulesError("Nobody can rest in the middle of a fight.")
    party = get_party(state)
    if not party:
        raise RulesError("Nobody is in the party yet.")
    targets = [_find_character(party, player)] if player else list(party.values())
    events: List[Event] = []
    for character in targets:
        if kind == "long":
            updated, more = combat.long_rest(character)
        else:
            updated, more = combat.short_rest(character, hit_dice, rng=rng)
        party = _store_party(party, updated)
        events.extend(more)
    return ToolResult(text=_lines(events), update={"party": put_party(party)}, events=events)
