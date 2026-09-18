# The Dungeon Master's tools

`src/tools/` is the layer between the engine and the model. Each tool is a
plain function over `GameState` that returns a `ToolResult`:

| Field | Holds |
|---|---|
| `text` | What the model reads — a few lines it narrates from, never a number it must compute |
| `update` | The state delta to write (`party`, `encounter`, `pending` as JSON), empty for read-only tools |
| `events` | The engine's `Event`s, for the log |
| `ok` | `False` when the call was refused or invalid; `text` then says why and what is possible |

Arguments are pydantic models (`AttackArgs`, `RequestCheckArgs`, …) with a
description per field. The same model validates a call **and** becomes the
tool's schema when it is bound to the model in PR-18, so there is one source
of truth for what a tool takes.

**`run_tool(state, name, args, rng=None)` never raises.** An unknown tool, a
bad argument, a rule the engine refuses ("It is Dorn's turn, not Kara's"), a
monster that does not exist — each comes back as a `ToolResult` with a
sentence the model can act on. A small model asked to fix a mistake needs the
mistake spelled out; a traceback teaches it nothing.

## The tools

| Tool | Does | Writes |
|---|---|---|
| `get_scene` | Party sheet lines, the encounter sheet if fighting, any pending roll | — |
| `lookup_monster(name)` | An SRD stat block summary; fuzzy on the name | — |
| `lookup_rules(question)` | The closest SRD passages, labelled — retrieval only, no generation, no rewrite | — |
| `request_check(player, ability, skill?, dc, reason?)` | Records a `PendingCheck`; the DC is never in the text; condition disadvantage is carried | `pending` |
| `request_save(player, ability, dc, reason?)` | Same for a save; an automatically failed save resolves at once | `pending` |
| `resolve_check(d20?)` | Resolves the pending roll with the player's die, or rolls for them. **Not offered to the model** — the player's `/roll` calls it | `pending` |
| `start_encounter(monsters)` | Numbers the monsters (`goblin-1`, `goblin-2`), rolls initiative, returns the sheet | `encounter` |
| `attack(attacker, target, attack_name?, mode?)` | One attack on the attacker's turn | `encounter`, and `party` when the fight ends |
| `end_turn()` | Next combatant; rolls a downed character's death save on the way | `encounter` |
| `end_encounter(outcome?)` | The party fled, the enemy yielded | `encounter` → `None`, `party` synced |
| `apply_damage(target, amount, type?)` / `heal(target, amount)` | In a fight, on the combatant; otherwise on the sheet | `encounter` or `party` |
| `apply_condition` / `remove_condition(target, condition)` | Same routing | `encounter` or `party` |
| `rest(kind, player?, hit_dice?)` | Short or long rest, one character or all; refused mid-fight | `party` |

When a fight ends — by the last blow, by the party going down, or by
`end_encounter` — the tool that ended it syncs HP, temp HP, conditions and
deaths back to the sheets and clears `encounter`, so the sheet is the source
of truth again the moment combat stops.

## Determinism

Every tool takes `rng`. The tests script it; the game will pass a
`random.Random` seeded per campaign so a session can be replayed from its log.
