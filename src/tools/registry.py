"""The tool registry, and the one call that never raises.

``run_tool(state, name, args)`` validates ``args`` against the tool's pydantic
model, runs it, and turns every failure — an unknown tool, a bad argument, a
rule the engine refuses to break, a monster that does not exist — into a
``ToolResult`` whose ``text`` says what went wrong and what is possible
instead. A small model asked to fix a mistake needs the mistake spelled out;
a traceback teaches it nothing.

``for_model`` marks the tools the DM is offered. ``resolve_check`` is not one
of them: the *player* resolves a pending roll, through ``/roll``.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Type

from pydantic import BaseModel, ValidationError

from src.engine.combat import RulesError
from src.srd import UnknownEntry
from src.tools import tools as t
from src.tools.tools import ToolResult
from src.utils.dice import RandomSource


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str  # what the model reads when choosing
    args: Type[BaseModel]
    fn: Callable[..., ToolResult]
    for_model: bool = True


TOOLS: Dict[str, ToolSpec] = {
    spec.name: spec
    for spec in [
        ToolSpec("get_scene", "The party's state, the fight if there is one, and any roll being waited on. Call this before deciding what happens.", t.NoArgs, t.get_scene),
        ToolSpec("lookup_monster", "An SRD creature's stat block: AC, HP, attacks, traits.", t.LookupMonsterArgs, t.lookup_monster),
        ToolSpec("lookup_rules", "The SRD passages that answer a rules question. Read them and answer in your own words, citing the label.", t.LookupRulesArgs, t.lookup_rules),
        ToolSpec("request_check", "Ask a player for an ability or skill check against a DC you set. Do not narrate the outcome until the roll comes back.", t.RequestCheckArgs, t.request_check),
        ToolSpec("request_save", "Ask a player for a saving throw against a DC you set. Do not narrate the outcome until the roll comes back.", t.RequestSaveArgs, t.request_save),
        ToolSpec("resolve_check", "Resolve the roll the DM is waiting on. Used by the player's /roll.", t.ResolveCheckArgs, t.resolve_check, for_model=False),
        ToolSpec("start_encounter", "Begin combat: roll initiative for the party and the named monsters.", t.StartEncounterArgs, t.start_encounter),
        ToolSpec("attack", "Resolve one weapon attack on the attacker's turn: to-hit, damage, and its effect.", t.AttackArgs, t.attack),
        ToolSpec("end_turn", "The current combatant's turn is over; advance to the next.", t.EndTurnArgs, t.end_turn),
        ToolSpec("end_encounter", "Stop the fight without a last blow: the party fled, the enemy yielded.", t.EndEncounterArgs, t.end_encounter),
        ToolSpec("apply_damage", "Damage from something other than an attack roll: a trap, a fall, a spell effect.", t.ApplyDamageArgs, t.apply_damage),
        ToolSpec("heal", "Restore hit points to a creature.", t.HealArgs, t.heal),
        ToolSpec("apply_condition", "Put a condition (prone, poisoned, restrained...) on a creature.", t.ConditionArgs, t.apply_condition),
        ToolSpec("remove_condition", "Lift a condition from a creature.", t.ConditionArgs, t.remove_condition),
        ToolSpec("rest", "A short rest (spend hit dice) or a long rest (full recovery), for one character or the party.", t.RestArgs, t.rest),
    ]
}


def model_tools() -> List[ToolSpec]:
    return [spec for spec in TOOLS.values() if spec.for_model]


def _explain_validation(name: str, error: ValidationError) -> str:
    problems = []
    for issue in error.errors():
        where = ".".join(str(p) for p in issue.get("loc", ())) or "arguments"
        problems.append(f"{where}: {issue.get('msg', 'invalid')}")
    expected = ", ".join(TOOLS[name].args.model_fields) or "no arguments"
    return f"Cannot run {name}: " + "; ".join(problems) + f". It takes: {expected}."


def run_tool(
    state: Mapping[str, Any],
    name: str,
    args: Optional[Mapping[str, Any]] = None,
    rng: Optional[RandomSource] = None,
    **injected: Any,
) -> ToolResult:
    """Validate, run, and explain instead of raising. Never throws."""
    spec = TOOLS.get(name)
    if spec is None:
        options = ", ".join(s.name for s in model_tools())
        return ToolResult(text=f"There is no tool called {name!r}. The tools are: {options}.", ok=False)

    try:
        parsed = spec.args.model_validate(dict(args or {}))
    except ValidationError as error:
        return ToolResult(text=_explain_validation(name, error), ok=False)

    try:
        return spec.fn(state, **parsed.model_dump(), rng=rng, **injected)
    except (RulesError, UnknownEntry, KeyError, ValueError) as error:
        message = str(error).strip("'\"")
        return ToolResult(text=f"Cannot {name.replace('_', ' ')}: {message}", ok=False)
    except Exception as error:  # a bug, not a rule — still no traceback to the model
        return ToolResult(text=f"{name} failed unexpectedly ({type(error).__name__}: {error}). Narrate around it.", ok=False)
