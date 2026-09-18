"""Where a turn goes, decided in code.

This replaces the LLM router. With a Discord front end (and this REPL), intent
mostly arrives labelled: `/rules` is a rules question, `/roll` is a roll,
`/join` is party management. What is left — free text — is play, and the
Dungeon Master handles play. A model was asked to classify these for ~2.7 s a
turn and got 12/12 on the benchmark; code gets 12/12 for free.

Deterministic dice survive here too: a message that is plainly a dice request
(`2d6+3`, `roll a d20 with advantage`) is rolled by the engine and answered
without a model, exactly as the old dice agent did on its common path.

Everything this module produces for the player is an `AIMessage` named
`intake`, so the front end renders it like any other reply and the DM sees
it in its context like anything else said at the table.
"""

import re
from typing import Any, Dict, List, Literal, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END
from langgraph.types import Command

from src.engine.pregens import PREGENS, describe_pregens, pregen
from src.graph.game_state import GameState, get_encounter, get_party, get_pending, put_party
from src.tools import run_tool
from src.utils.dice import DiceRoller, RandomSource

SPEAKER = "intake"

# --- dice in free text ---------------------------------------------------------------

DICE_EXPRESSION = re.compile(r"\d*d\d+(?:\s*[+-]\s*(?:\d*d\d+|\d+))*", re.IGNORECASE)
DICE_NOTATION = re.compile(r"\b\d*d\d+\b", re.IGNORECASE)
FLAT_MODIFIER = re.compile(r"([+-])\s*(\d+)(?!\s*d\d)", re.IGNORECASE)
ROLL_VERB = re.compile(r"\broll(s|ed|ing)?\b", re.IGNORECASE)
ADVANTAGE = re.compile(r"\badvantage\b", re.IGNORECASE)
DISADVANTAGE = re.compile(r"\bdisadvantage\b", re.IGNORECASE)
PURPOSE = re.compile(r"\b(?:for|to check|to see if)\b\s+(.+)$", re.IGNORECASE)
# "what does d20 mean?" contains notation but is a question for the table.
QUESTION_OPENER = re.compile(
    r"^\s*(what|how|why|when|where|which|who|whose|does|do|did|is|are|was|were|"
    r"can|could|should|would|explain|describe|tell\s+me)\b",
    re.IGNORECASE,
)
BARE_NOTATION = re.compile(r"^[\dd\s+\-]+$", re.IGNORECASE)


def extract_dice_expression(message: str) -> Tuple[Optional[str], int]:
    """`(notation, modifier)` read literally from the text, or `(None, 0)`."""
    match = DICE_EXPRESSION.search(message or "")
    if not match:
        return None, 0
    expression = match.group(0).replace(" ", "")
    terms = []
    for sign, term in re.findall(r"([+-]?)(\d*d\d+)", expression, re.IGNORECASE):
        if sign == "-":
            raise ValueError(f"cannot subtract dice: {sign}{term} in {expression!r}")
        terms.append(term if term[0].lower() != "d" else f"1{term}")
    if not terms:
        return None, 0
    modifier = sum(int(f"{sign}{value}") for sign, value in FLAT_MODIFIER.findall(expression))
    return "+".join(terms), modifier


def is_dice_request(text: str) -> bool:
    """Unambiguously a request to roll: bare notation, or notation with a roll verb."""
    text = (text or "").strip()
    if not text or QUESTION_OPENER.match(text) or not DICE_NOTATION.search(text):
        return False
    return bool(BARE_NOTATION.match(text) or ROLL_VERB.search(text))


def roll_text(text: str, rng: Optional[RandomSource] = None) -> str:
    """Roll what the text asks for and describe it. Raises ValueError on junk."""
    notation, modifier = extract_dice_expression(text)
    if notation is None:
        raise ValueError(f"no dice in {text!r}")
    disadvantage = bool(DISADVANTAGE.search(text))
    advantage = bool(ADVANTAGE.search(text)) and not disadvantage
    purpose = PURPOSE.search(text)
    description = purpose.group(1).strip().rstrip(".") if purpose else ""
    if description.lower().startswith(("advantage", "disadvantage")):
        description = ""
    modifier_text = f" + {modifier}" if modifier > 0 else (f" - {abs(modifier)}" if modifier < 0 else "")
    for_text = f" for {description}" if description else ""

    if advantage or disadvantage:
        first = DiceRoller.roll_multiple(notation, rng)
        second = DiceRoller.roll_multiple(notation, rng)
        a, b = sum(r.total for r in first), sum(r.total for r in second)
        kept = max(a, b) if advantage else min(a, b)
        word = "advantage" if advantage else "disadvantage"
        return (
            f"🎲 Rolled {notation} with {word}{modifier_text}{for_text}: **{kept + modifier}** "
            f"(rolls {a} and {b}, took the {'higher' if advantage else 'lower'})"
        )

    rolls = DiceRoller.roll_multiple(notation, rng)
    total = sum(r.total for r in rolls) + modifier
    detail = ", ".join(str(r) for r in rolls)
    return f"🎲 Rolled {notation}{modifier_text}{for_text}: **{total}** ({detail})"


# --- attacks declared in play, during a fight -----------------------------------------------
#
# Measured on qwen2.5:7b (PR-18): asked "I swing my longsword at the goblin" with
# the `attack` tool bound, the model narrated two hits and never called it —
# the goblin stood at 7/7 HP while the prose had it bleeding. Whether an attack
# *happened* is not the model's to decide any more than the damage is. So a
# player's attack on their turn, at a monster the text names (or the only one
# standing), is resolved here through the engine, and the DM is handed the
# result to narrate. Anything ambiguous — no target, two goblins and "the
# goblin", a negation — goes to the DM, which still has the tool.

ATTACK_VERB = re.compile(
    r"\b(attack|attacks|swing|swings|strike|strikes|stab|stabs|slash|slashes|"
    r"shoot|shoots|fire|fires|hit|hits|lunge|lunges|charge|charges|cut|cuts|smash|smashes|"
    r"throw|throws|hurl|hurls|loose|looses)\b",
    re.IGNORECASE,
)
NEGATION = re.compile(r"\b(don't|do not|won't|will not|not|instead of|rather than|no longer)\b", re.IGNORECASE)


def _named_target(encounter, text: str):
    """The one living monster the text points at, or None."""
    lowered = text.lower()
    alive = [m for m in encounter.monsters if not m.dead]
    by_id = [m for m in alive if m.id.lower() in lowered]
    if len(by_id) == 1:
        return by_id[0]
    if by_id:
        return None  # "goblin-1 and goblin-2": let the DM sort it out
    by_name = [m for m in alive if m.name.lower() in lowered]
    if len(by_name) == 1:
        return by_name[0]
    if not by_name and len(alive) == 1:
        return alive[0]  # "I attack it": only one thing to attack
    return None


def _named_weapon(character, text: str) -> Optional[str]:
    lowered = text.lower()
    for weapon in character.weapons:
        if weapon.name.lower() in lowered:
            return weapon.name
    return None


def prefilter_attack(state: GameState, text: str, rng: Optional[RandomSource]) -> Optional[Command]:
    """Only during a fight. Out of one, `intake` cannot know whether "the
    goblin" is real, dead, or a figure of speech — measured, it started a
    second fight against a goblin the party had just killed. The DM decides
    when a fight begins; the `attack` tool starts one when it is called."""
    if QUESTION_OPENER.match(text) or NEGATION.search(text) or not ATTACK_VERB.search(text):
        return None
    encounter = get_encounter(state)
    if encounter is None or not encounter.active:
        return None
    actor = encounter.get(encounter.current)
    if actor.kind != "character":
        return None
    target = _named_target(encounter, text)
    if target is None:
        return None
    character = get_party(state).get(actor.name)
    weapon = _named_weapon(character, text) if character else None
    args = {"attacker": actor.id, "target": target.id}
    if weapon:
        args["attack_name"] = weapon
    result = run_tool(state, "attack", args, rng=rng)
    if not result.ok:
        return None  # the tool said why; let the DM read that
    return Command(
        goto="dungeon_master",
        update={
            "messages": [AIMessage(content=result.text, name=SPEAKER)],
            **result.update,
            "tool_steps": 0,
        },
    )


# --- commands ----------------------------------------------------------------------------

COMMAND = re.compile(r"^\s*/(\w+)\s*(.*)$", re.DOTALL)

HELP = """Commands:
  /join <fighter|rogue|cleric|wizard|ranger|barbarian> [as <name>]  — take a pregenerated character
  /party        — who is at the table, and how they are doing
  /roll [d20]   — resolve the roll the DM asked you for (or roll dice: /roll 2d6+3)
  /rules <q>    — ask the rules assistant, with citations from the SRD
  /help         — this
Anything else is play: say what your character does."""


def parse_command(text: str) -> Optional[Tuple[str, str]]:
    match = COMMAND.match(text or "")
    if not match:
        return None
    return match.group(1).lower(), match.group(2).strip()


def _reply(text: str, **update: Any) -> Command:
    return Command(goto=END, update={"messages": [AIMessage(content=text, name=SPEAKER)], "last_response": text, **update})


def _speaker_id(state: GameState) -> str:
    for message in reversed(list(state.get("messages") or [])):
        if isinstance(message, HumanMessage):
            return getattr(message, "name", None) or "player"
    return "player"


def _join(state: GameState, argument: str) -> Command:
    match = re.match(r"^(\w+)(?:\s+as\s+(.+))?$", argument.strip(), re.IGNORECASE)
    if not argument or not match:
        return _reply("Usage: /join <pregen> [as <name>]\n" + "\n".join(describe_pregens()))
    key, name = match.group(1).lower(), (match.group(2) or "").strip()
    if key not in PREGENS:
        return _reply(f"No pregen called {key!r}. Choose one of: {', '.join(PREGENS)}.")
    party = get_party(state)
    character = pregen(key, player_id=_speaker_id(state), name=name or None)
    if any(c.name.lower() == character.name.lower() for c in party.values()):
        return _reply(f"There is already a {character.name} at the table. Pick another name: /join {key} as <name>.")
    party[character.name] = character
    return _reply(
        f"{character.name} joins the party — {character.sheet_line()}.",
        party=put_party(party),
    )


def _roll(state: GameState, argument: str, rng: Optional[RandomSource]) -> Command:
    if state.get("pending"):
        d20 = None
        if argument:
            if not argument.isdigit() or not 1 <= int(argument) <= 20:
                return _reply("Give me the d20 you rolled (1–20), or just /roll to let me roll it.")
            d20 = int(argument)
        result = run_tool(state, "resolve_check", {"d20": d20}, rng=rng)
        # The DM narrates the outcome, with the result in front of it.
        return Command(
            goto="dungeon_master",
            update={
                "messages": [AIMessage(content=result.text, name=SPEAKER)],
                "pending": None,
                "tool_steps": 0,
                "current_task": f"[roll resolved] {result.text}",
            },
        )
    if not argument:
        return _reply("Nobody has asked you for a roll. To roll dice anyway: /roll 2d6+3.")
    try:
        return _reply(roll_text(argument, rng))
    except ValueError as exc:
        return _reply(f"I could not read that as dice ({exc}). Try /roll 1d20+5.")


def intake(state: GameState, rng: Optional[RandomSource] = None) -> Command[Literal["dungeon_master", "researcher", "__end__"]]:
    """Classify the turn and send it on. No model call, ever."""
    text = (state.get("current_task") or "").strip()
    if not text:
        for message in reversed(list(state.get("messages") or [])):
            if isinstance(message, HumanMessage):
                text = str(message.content).strip()
                break

    command = parse_command(text)
    if command:
        name, argument = command
        if name == "rules":
            if not argument:
                return _reply("Ask something: /rules how does grappling work")
            return Command(goto="researcher", update={"current_task": argument})
        if name == "roll":
            return _roll(state, argument, rng)
        if name == "join":
            return _join(state, argument)
        if name in ("party", "scene", "sheet"):
            return _reply(run_tool(state, "get_scene").text)
        if name == "help":
            return _reply(HELP)
        return _reply(f"Unknown command /{name}.\n{HELP}")

    # A pending roll answered with a bare number.
    if state.get("pending") and text.isdigit() and 1 <= int(text) <= 20:
        return _roll(state, text, rng)

    # A pending roll, and the player says something else: the table waits.
    # Measured on qwen2.5:7b, sending this to the DM in narrate-only mode
    # produced a pseudo tool call in prose; a sentence from code is better.
    pending = get_pending(state)
    if pending is not None and not is_dice_request(text):
        return _reply(
            f"The DM is waiting on {pending.player} for a {pending.label}"
            f"{' — ' + pending.reason if pending.reason else ''}. "
            f"Roll it with /roll <d20>, or /roll to have it rolled for you."
        )

    if is_dice_request(text):
        try:
            return _reply(roll_text(text, rng))
        except ValueError as exc:
            return _reply(f"I could not read that as dice ({exc}). Try 2d6+3.")

    attacked = prefilter_attack(state, text, rng)
    if attacked is not None:
        return attacked

    return Command(goto="dungeon_master", update={"tool_steps": 0})
