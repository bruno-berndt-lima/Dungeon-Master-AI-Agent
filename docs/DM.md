# The Dungeon Master

Since PR-18 the graph has one model-driven agent, the Dungeon Master, and one
model-driven helper, the researcher. Routing is code. This file describes the
DM; `docs/TOOLS.md` describes what it can do and `docs/ENGINE.md` what decides
the outcomes.

## The turn

```
intake ──▶ dungeon_master ──(tool calls)──▶ tools ──▶ dungeon_master ──▶ … ──▶ END
   │              └──(no tool calls: narration)──────────────────────────────▶ END
   ├─▶ researcher ─▶ END      /rules <question>
   └─▶ END                    /roll, /join, /party, /help, bare dice
```

1. **`intake`** (`src/graph/intake.py`) classifies the turn in code: a slash
   command, a bare dice request, an answer to a pending roll, a declared
   attack during a fight, or play. Play goes to the DM with a fresh tool
   budget. No model call, ever.

   **A declared attack is resolved before the DM is called.** Measured on
   `qwen2.5:7b`: "I swing my longsword at the goblin" with `attack` bound
   produced two narrated hits and no tool call, and a goblin at full HP.
   Whether an attack *happened* is not the model's to decide any more than
   the damage is. So on a player's turn, an attack verb plus a living monster
   the text identifies (by id, by name, or the only one standing) runs the
   `attack` tool from `intake`, and the DM narrates the result it is handed.
   Ambiguity — two goblins and "the goblin", a negation, a question — goes to
   the DM, which still has the tool. **Out of a fight, `intake` never starts
   one**: it cannot know whether "the goblin" is real, dead, or a figure of
   speech (measured: it began a second fight against a goblin the party had
   just killed). The DM decides when a fight begins — and `attack` starts one
   by itself when it is called with none under way, with the attacker going
   first, because a refusal that expects a second tool call is one a 7B does
   not recover from (measured: it narrated instead).
2. **`dungeon_master`** (`DungeonMaster.process_task`) builds the prompt —
   `DUNGEON_MASTER_PROMPT`, then **the table right now** (`get_scene`: party
   lines, the encounter sheet, any pending roll), then the last six narrative
   messages and this turn's tool exchange — and calls the model with the tools
   bound. Tool calls → `tools`. Otherwise the reply is the narration and the
   turn ends.
3. **`tools`** (`DungeonMaster.run_tools`) runs every requested tool in order
   through `run_tool`, each seeing the previous one's effect, writes the state
   deltas, and returns a `ToolMessage` per call. Back to the DM.

**The loop is capped** at `MAX_TOOL_STEPS = 4` per turn. Past the cap the DM is
called *without* tools and told to narrate what it has. A turn always ends with
narration; an empty reply becomes a fixed line rather than silence.

**Narrate-only mode** also applies while a roll is pending: the DM asked a
player for a check, and until `/roll` resolves it the DM is not offered tools.
In practice the DM rarely sees that case — `intake` answers play text with a
reminder of the owed roll, in code, because a 7B model told it could not call
tools wrote a pseudo tool call in prose instead.

## Requested rolls

`request_check` / `request_save` record a `PendingCheck` in state — with the
DC, which the model is told never to say. The DM's turn ends by asking for the
roll. The player's `/roll 15` (or `/roll` to have it rolled, or a bare `15`)
goes through `intake`, which resolves the check against the sheet
(`resolve_check`, a tool the model is not offered), posts the result as an
`intake` message, and sends the turn to the DM to narrate the outcome.

This was specified as a LangGraph `interrupt()`. A pending field plus a new
turn does the same job without holding a thread open — which matters once a
Discord channel has several players who may act while one of them finds their
dice.

## What the model may and may not do

The prompt is a contract, not a mood board:

- It never produces a number. HP, AC, damage, DCs and rolls exist only once a
  tool returns them. The tools describe every arithmetic step.
- Uncertain outcome → `request_check` / `request_save`, then stop and ask.
- A fight → `start_encounter`, then `attack` for a player's attack. The
  monsters act on their own after every player's turn (see `docs/TOOLS.md`);
  the model is never asked to run them.
- A refused tool explains itself; the DM does what it says or narrates around it.

`tests/test_dm_loop.py` drives the whole graph with a scripted stand-in for the
model: the loop, the cap, narrate-only mode, a refused tool, a pending roll
resolved by the player, and the two-goblin fight from PR-16 played to its end
through the graph. None of it needs a daemon.

## Context and cost

Prompt-eval dominates time-to-first-token on CPU, so the prompt is kept small:
the scene sheet (~150 tokens for a four-creature fight) replaces most of the
transcript, `CONTEXT_WINDOW = 6` prior narrative messages are kept, and the
tool traffic of earlier turns is dropped from the context (it is in state for
the log). PR-19 adds the rolling summary; PR-20 measures models on tool choice.

Every model call and every tool call is logged through `_log_interaction` with
a `stage` of `plan`, `tool`, or `narrate`, so a turn can be reconstructed from
`logs/llm_interactions/`.

## The researcher

Unchanged in mechanism: `/rules <question>` goes straight to `ResearcherAgent`,
which retrieves from the SRD index, rewrites the question once if the score is
low, answers, and appends the passages consulted. It now reads the bare
question from `current_task`, which `intake` sets. The DM has `lookup_rules`
for the same index, retrieval only — it answers in its own voice.

## What went

- **`GameSupervisor`** and `SUPERVISOR_PROMPT` — a model call per turn to
  answer a question the front end already labels.
- **`DiceRollerAgent`**, `DICE_PARSE_PROMPT`, `DICE_ROLLER_PROMPT` — bare dice
  are rolled by `intake` through the engine; everything else a roll is *for* is
  a `request_check`.
- **Scene extraction** (`SCENE_EXTRACTION_PROMPT`, `game_state`) — the world
  state was derived from prose and invented items. The engine's state is the
  truth now; the narration is derived from it.
