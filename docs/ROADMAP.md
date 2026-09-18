# ROADMAP — from rules chatbot to multiplayer Dungeon Master

Written 2026-09-18 after a full read of the repository, the test suite, the
interaction logs, and the installed stack. `docs/SPECS.md` was the contract for
the refactor; every PR in it has landed. This document is the contract for
what comes next.

Decisions this plan is built on (confirmed with Bruno, 2026-09-18):

| Decision | Choice |
|---|---|
| Where the game is played | **Discord bot**. One channel = one campaign. |
| Hardware | An **Apple Silicon M4** over Tailscale (`OLLAMA_HOST`) as the main model host; this **Intel i9** stays the fallback and must keep working. |
| Mechanics | **Faithful to 5e.** Code resolves checks, attacks, HP, initiative, conditions. The model narrates and chooses; it never decides a number. |
| Purpose | Playable with friends **and** a showcase. Keep the bar: one spec per PR, offline tests, docs that match the code. |
| Models | **Local, via Ollama.** No provider swap. |

---

## 1. The idea, restated

An AI Dungeon Master that runs a Dungeons & Dragons 5e adventure for a **party
of human players** in a Discord channel. Players describe what their characters
do; the DM narrates the world's response, asks for the rolls the rules require,
resolves them against the party's character sheets and the monsters' stat
blocks, and keeps the campaign's state across sessions. Rules answers are
grounded in the SRD 5.1 and cite the passage. Everything runs on local models.

The current code is the first third of that: a **single-player rules chatbot
with a narrator**. This plan is about the other two thirds — a game engine, and
a party.

---

## 2. Where the project stands

### What is done well and carries forward as-is

- **The model layer** (`src/models/llm.py`). One factory, per-role model map,
  `DND_MODEL_*` / `OLLAMA_HOST` overrides, legible errors for a dead daemon or an
  unpulled model. The Tailscale setup is literally one environment variable.
- **Deterministic dice** (`src/utils/dice.py`, the regex extraction in
  `dice_roller.py`). The lesson learned there — *a formal language is read by a
  parser, not a model* — is the design principle of the whole engine below.
- **The SRD corpus and ingestion** (`corpus/srd/`, `srd_loader.py`,
  `scripts/ingest.py`). One document per entry, chunks re-headed with the entry
  name, citations by name. Reproducible from a clean clone in 35 s. This is
  better than most RAG-over-rulebooks projects, and the JSON is also a
  **structured bestiary** — `Monsters.json` carries `armor_class`, `hit_points`,
  `attack_bonus`, `damage_dice` — which the combat engine will read directly.
- **Retrieval with a free relevance gate and one rewrite** (`researcher.py`).
  Measured, not guessed. Stays as the `/rules` command and becomes a DM tool.
- **Persistence.** `SqliteSaver` is wired; a Discord channel id maps 1:1 onto a
  `thread_id`. `AsyncSqliteSaver` is installed and imports.
- **Streaming** through `stream_mode="messages"` with node/tag filtering.
- **Test discipline.** 217 tests, none needing a model, all green on 2026-09-18.
- **Docs.** `KNOWN_ISSUES.md` with a status ledger, measured numbers everywhere.

Roughly 70% of the source survives untouched. What gets replaced is small in
lines and large in shape.

### What does not fit the goal

1. **The graph topology is a help desk, not a game loop.** Supervisor → one of
   three workers → END. One routing decision, one worker, turn over. A real
   turn is a *composition*: "I attack the goblin" needs the DM to decide an
   attack roll is required, roll it, compare to the goblin's AC, apply damage,
   and narrate. Today those are three unrelated user turns and nothing links
   them. The supervisor pattern was borrowed from the LangGraph multi-agent
   tutorial; it fits routing support tickets, not adjudicating combat.

2. **World state is derived from prose, backwards.** `dungeon_master.py`
   narrates first and then asks a model to *extract* `location / inventory /
   effects` from the narration (~5 s per turn). The 2026-07-31 log shows it
   inventing items (`"dagger"` gained when the player already had one,
   `"a torch"` from atmosphere). For a faithful game the arrow must flip:
   **engine state is the source of truth, the model reads it.**

3. **No notion of who is speaking.** `HumanMessage` carries no name; one
   thread; `players`, `npcs`, `turn_order`, `current_speaker` are declared in
   `GameState` and never written. Those unfilled fields are exactly the
   multiplayer model, waiting.

4. **`src/actors/` has the right instinct and the wrong shape.**
   `NPC.process_message` with an LLM personality is a rabbit hole. NPCs are
   stat blocks plus the DM's voice. Delete the package; the engine replaces it.

5. **LLM routing costs ~2.7 s per turn to answer a question that mostly has a
   deterministic answer.** With Discord, intent arrives labelled: `/roll`,
   `/rules`, `/sheet` are commands; everything else is play. The prefilter idea
   generalises; the supervisor model call does not need to exist.

6. **The researcher is a destination when it should also be a tool.** Players
   asking rules is a feature (keep it). The DM looking up "what does a goblin
   do on its turn" mid-narration is a tool call, and the SRD JSON answers most
   of those without embeddings at all.

### Verdict

The direction was right and the execution is clean; the *architecture* was
sized for a rules chatbot and has reached the end of what it can express. The
next step is not another agent. It is an engine, and a DM that uses it.

---

## 3. Stack review

Checked against PyPI and the installed venv on 2026-09-18.

| Component | Installed | Latest | Verdict |
|---|---|---|---|
| `langgraph` | 1.2.10 | 1.2.11 | **Keep.** Bump. |
| `langchain-core` | 1.5.3 | 1.6.3 | Keep. Bump. |
| `langchain` (meta) | 1.3.14 | 1.4.2 | Not imported anywhere — only `langchain_core`, `langchain_text_splitters`, `langchain_ollama`, `langchain_chroma`. Drop it. |
| `langchain-community` | 0.4.2 | 0.4.2 | **Sunset upstream** (deprecation warning on every test run). One import, `PyMuPDFLoader`. Replace with ten lines over `pymupdf`. |
| `langchain-ollama` | 1.1.0 | 1.1.0 | Keep. Structured output, `bind_tools`, both verified on installed models. |
| `chromadb` / `langchain-chroma` | 1.5.9 / 1.1.0 | same | Keep. 3,082 chunks is tiny; nothing else is warranted. |
| `sentence-transformers` + `torch` + `transformers` + `numpy` pins | — | — | **Replace with Ollama embeddings** (`all-minilm` is the same `all-MiniLM-L6-v2`, 384-dim; `nomic-embed-text` is the stronger option). This deletes the entire Intel-macOS pin block, the `transformers` and `numpy` pins, and the second reason for "Python 3.12 exactly". |
| Ollama | 0.32.5 | — | Keep. Native JSON-schema `format`, tool calling, `think` toggle. |
| `discord.py` | not installed | 2.x | **Add.** Async, slash commands, the standard. |
| `pydantic-ai` | — | 2.45 | Evaluated, **not adopted** — see below. |

### Why LangGraph stays

The question was asked honestly: is there a better harness in 2026? The
credible alternatives are Pydantic AI (typed, light, excellent for a single
agent with tools), the OpenAI Agents SDK and Google ADK (vendor-centric; nothing
for an Ollama project), and CrewAI (role-play orchestration, wrong fit). What a
multi-session, multi-player campaign actually needs from a harness is:

- **Durable state per campaign** across process restarts — LangGraph's
  checkpointer, already wired, keyed by channel.
- **Pausing a turn for human input** — `interrupt()` / `Command(resume=...)`,
  which is how "the DM asks Kara for a Stealth check" waits for Kara.
- **A bounded tool loop** — `ToolNode` plus a step cap.
- **Token streaming with provenance** — already working.

Pydantic AI does the tool loop better and the rest not at all; pairing it with
LangGraph means two frameworks. Rewriting to save ~200 lines of graph code and
lose the checkpointer is a bad trade. **Keep LangGraph, use less of LangChain**:
drop the meta package, drop community, keep `langchain-core` messages and
`langchain-ollama`.

### Models

`qwen2.5:7b` was chosen in July 2026 on a 12-case routing benchmark. Routing is
going away, so that number no longer picks the model. The job that matters now
is **choosing the right tool with the right arguments, then narrating well**.
Candidates to measure (PR-20 builds the harness; do not switch on reputation):

| Role | Current | Candidate | Note |
|---|---|---|---|
| `dungeon_master` | `qwen2.5:7b` | `qwen3:8b` (thinking **off**), `gemma3:12b` on the M4 | Tool-call accuracy and prose quality both matter. |
| `researcher` | `qwen2.5:7b` | same as DM | Grounded answering; keep it on the same model to avoid a second resident. |
| `dice_roller` | `llama3.2:3b` | — | Only reached when a request names no dice; may be deleted with the supervisor. |
| embeddings | `all-MiniLM-L6-v2` via torch | `nomic-embed-text` via Ollama | Re-index, re-measure `RELEVANCE_THRESHOLD`. |

On the M4 (unified memory, Metal) a 12–14B model is comfortable. On the Intel
box every generation is 5 tok/s and a tool-using turn is 2–4 generations, so
the design rules below exist to keep the Intel path playable, not just possible.

---

## 4. Target architecture

```
Discord channel (one campaign)
   │  "Kara: I sneak toward the guard"      /roll stealth      /rules grapple
   ▼
bot.py (discord.py, async)
   │  thread_id = channel id · one asyncio.Lock per channel · chunked replies
   ▼
LangGraph  ──▶ intake ──▶ dm ⇄ tools ──▶ END        (bounded: ≤ 4 tool calls)
               │  deterministic: command? roll result? free text?
               │  /rules → researcher (unchanged RAG)  ·  /roll → engine, no model
               ▼
            src/engine/   pure Python, no LLM, fully unit-tested
               characters · checks · attacks · combat tracker · conditions · rests
               ▲
            src/srd/      structured access to corpus/srd/*.json (monsters, spells, gear)
```

**Three rules that keep it playable on CPU and honest as a game:**

1. **The model never produces a number that matters.** Every roll, DC
   comparison, damage total, HP change, and turn order comes from `src/engine/`.
   The model calls a tool and reads the result. This is PR-05's lesson applied
   everywhere.
2. **Prompt size is a budget.** The DM sees a ~150-token *scene sheet* rendered
   by the engine (party HP/AC/conditions, who is up in combat, location), a
   rolling summary of the campaign, and the last few messages — not the
   transcript. Time-to-first-token is prompt-eval bound on CPU (KNOWN_ISSUES
   #26); this is the lever.
3. **Every turn ends with visible output in bounded time.** Tool loop capped;
   tool errors are returned as text the model can read, never raised; a turn
   that hits the cap narrates what it has.

**How a check flows** (the interaction that does not exist today):

```
Kara: "I sneak past the guard"
  dm → tool request_check(player="Kara", ability="DEX", skill="Stealth", dc=13)
     → engine records a pending check; graph interrupts
  bot: "Kara — give me a Dexterity (Stealth) check."
Kara: /roll                      (or "/roll 1d20", or types "17")
  bot → engine resolves: d20 + Kara's Stealth modifier vs DC, advantage if any
     → graph resumes with a RollResult event
  dm narrates the outcome, may call apply_condition / advance_combat, ends
```

Two generations for the DM, zero model calls for the roll. Players roll their
own dice because that is the fun; `auto_roll` per campaign is a setting for
groups who would rather not wait.

---

## 5. Phases and PRs

Numbering continues from `docs/SPECS.md`. Same rules: one branch per PR, the
spec ID and verification in the PR body, `pytest` green before review, no
model calls in the suite. Sizes are relative (S ≈ an evening, M ≈ a weekend,
L ≈ two weekends).

```
Phase 0   {PR-11, PR-12, PR-13}                 housekeeping, parallel-safe
Phase 1   PR-14 → PR-15 → PR-16                 the engine (no LLM)
Phase 2   PR-17 → PR-18 → PR-19 → PR-20         the DM becomes a tool user
Phase 3   PR-21 → PR-22 → PR-23                 the party, then Discord
Phase 4   PR-24, PR-25                          spells, showcase
```

Phase 1 is deliberately model-free so it can be built and tested entirely on
the Intel machine, and so the engine is a showcase piece on its own.

### Phase 0 — Housekeeping

**PR-11 — Trim the LangChain surface (S)**
`chore/trim-langchain`. Replace `PyMuPDFLoader` with a direct `pymupdf` loader
in `src/data/loader.py`; drop `langchain` and `langchain-community` from
`requirements.txt`; bump `langgraph` and `langchain-core` to current. Acceptance:
no `langchain_community` import, no deprecation warning in `pytest`, suite green.

**PR-12 — Embeddings through Ollama (M)**
`refactor/ollama-embeddings`. `create_embeddings()` returns `OllamaEmbeddings`
(`nomic-embed-text`; `all-minilm` if the index should stay 384-dim). Delete the
`torch` / `transformers` / `numpy` / `sentence-transformers` /
`langchain-huggingface` pins and the Intel-macOS comment block. Rebuild the
index, re-measure the on/off-topic score bands, reset `RELEVANCE_THRESHOLD`,
and record the numbers in `RAG_PIPELINE.md`. Relax `requires-python` to
`>=3.11` and correct `CLAUDE.md`. Acceptance: fresh venv installs with no torch;
`scripts/ingest.py` builds; the three benchmark queries in `RAG_PIPELINE.md`
still land on the right entry.

**PR-13 — Resume a campaign from the CLI (S)**
`feat/cli-resume`. `main.py --thread <id>` and `--list`. Already noted in
`REFACTOR_NOTES.md` as "one flag away". Keeps the REPL useful as the
development harness through Phase 3.

### Phase 1 — The engine

**PR-14 — Characters and checks (M)**
`feat/engine-characters`. New package `src/engine/`:
- `character.py` — pydantic `Character`: identity (`player_id`, `name`), race,
  class, level, six abilities, proficiency bonus, skill and save proficiencies,
  AC, max/current/temp HP, hit dice, speed, conditions, inventory, equipped
  weapon(s). Derived values (modifiers, passive Perception) are properties.
  JSON round-trips, because it lives inside the checkpointed `GameState`.
- `checks.py` — `ability_check`, `saving_throw`, `attack_roll` over
  `src/utils/dice.py`: advantage/disadvantage, natural 1/20, crit damage
  doubling, `CheckResult` dataclass with the roll, modifier, total, DC, success.
- `pregens.py` — five or six SRD-only pregenerated level-1 characters (fighter,
  rogue, cleric, wizard, ranger…) as data, so `/join` needs no creation flow.
- Tests: every rule above, offline, seeded.
Acceptance: `Character.model_validate_json(c.model_dump_json()) == c`; a
Stealth check for the pregen rogue uses DEX + proficiency; crits double dice
and not the modifier.

**PR-15 — Structured SRD access (S)**
`feat/srd-data`. `src/srd/` reads `corpus/srd/*.json` once and exposes
`monster(name)`, `spell(name)`, `equipment(name)`, `condition(name)`, with fuzzy
name matching. `Combatant.from_monster(...)` builds an engine combatant from a
stat block — AC, HP (average or rolled from `hit_dice`), attacks with
`attack_bonus` and `damage_dice`. Acceptance: `monster("goblin")` yields AC 15,
HP 7, a scimitar at +4 for `1d6+2`; a misspelling within edit distance 2 still
resolves; an unknown name raises.

**PR-16 — Combat tracker (L)**
`feat/engine-combat`. `src/engine/combat.py`: `Encounter` with initiative
(rolled per combatant), round counter, `current()`, `advance()`, `attack(attacker,
target, weapon)` resolving to-hit vs AC and applying damage, `damage`/`heal`
with temp HP, conditions with a minimal effect table (prone, poisoned,
restrained, unconscious → advantage/disadvantage on the relevant rolls), death
saves at 0 HP for characters, monsters die at 0. `short_rest` / `long_rest`.
Everything returns an `Event` list (`AttackResolved`, `DamageApplied`,
`Dropped`…) the DM narrates from. Acceptance: a scripted goblin fight against
the pregen fighter runs to completion deterministically under a seed; every
event type is tested; no function in the package imports anything from
`src/agents` or `langchain`.

`GameState` changes in this phase: add `party: dict[str, Character]`,
`encounter: Encounter | None`, `pending: PendingCheck | None`, `summary: str`;
delete `players`, `npcs`, `current_speaker`, `turn_order`; delete
`src/actors/`. Closes KNOWN_ISSUES #14 and #16.

### Phase 2 — The DM becomes a tool user

**PR-17 — Tools (M)**
`feat/dm-tools`. `src/tools/` — thin, validated wrappers over the engine, each
returning a short string the model can read:
`get_scene()`, `request_check(player, ability, skill?, dc)`,
`request_save(player, ability, dc)`, `attack(attacker, target)`,
`apply_damage(target, amount, type)`, `apply_condition(target, condition)`,
`start_encounter(monsters=[...])`, `end_turn()`, `lookup_rules(question)`
(the existing retriever), `lookup_monster(name)`. Every tool validates
arguments with pydantic and returns an explanation instead of raising
(`"There is no combatant named 'Gobin'. Party: Kara, Dorn. Enemies: goblin-1."`).
Acceptance: every tool has an offline test with a fake state; no tool calls a
model; a bad argument never propagates an exception.

**PR-18 — New graph, no supervisor (L)**
`refactor/dm-tool-loop`. The pivot.
- `intake` node — pure code. Classifies the incoming event: slash command
  (already resolved by the bot/REPL), `RollResult`, or play text. Routes
  `/rules` to `researcher` unchanged; everything else to `dm`.
- `dm` node — `create_llm("dungeon_master").bind_tools([...])`, then a
  `ToolNode`, with a step counter in state and a hard cap of 4 tool calls per
  turn. `request_check` calls `interrupt()`; the bot resumes with the roll.
  Narration still streams by node name; tool-call chunks are tagged `internal`.
- Delete `GameSupervisor`, `SUPERVISOR_PROMPT`, `DiceRollerAgent` (dice notation
  in free text is handled by `intake` + engine; the "no dice named" fallback
  becomes `request_check`), and scene extraction. Rewrite
  `DUNGEON_MASTER_PROMPT` around the scene sheet and the tool contract ("you
  never state a result you did not get from a tool").
- `docs/AGENTS.md` becomes `docs/DM.md`; `ARCHITECTURE.md` redrawn.
Acceptance: the goblin fight from PR-16 plays end to end through the REPL with
a stub model that emits scripted tool calls (offline); the step cap is tested;
a turn where the model emits no tool call still narrates; `grep -rn supervisor
src/` is empty.

**PR-19 — Memory (M)**
`feat/campaign-memory`. Rolling summary: when `messages` exceeds N, an
`internal`-tagged call folds the oldest into `state["summary"]` and the
messages are trimmed with `RemoveMessage`. The scene sheet is rendered by the
engine (`Encounter.sheet()`, party one-liners) and prepended to the system
prompt. Measure time-to-first-token on both machines at the resulting prompt
size and record it. Acceptance: a 60-turn scripted session keeps the prompt
under a fixed token budget (test with a stub tokenizer); an early fact (a name
given in turn 2) survives to turn 60 through the summary.

**PR-20 — Evaluation harness and model choice (M)**
`feat/eval-harness`. `scripts/eval_dm.py` runs a fixed set of ~20 situations
(out of combat, in combat, a rules question inside play, a player trying to
declare a result) against a live daemon and scores: correct tool chosen,
arguments valid, no invented numbers in the narration (regex for dice totals /
HP claims), latency. Run `qwen2.5:7b`, `qwen3:8b` (think off), and one 12–14B
on the M4; record the table in `REFACTOR_NOTES.md`; set `AGENT_MODELS`
accordingly. Not part of `pytest`. Acceptance: the table exists with numbers
from both machines and the chosen defaults are justified by it.

### Phase 3 — The party, then Discord

**PR-21 — Multiple players, hot-seat (M)**
`feat/party`. Messages carry `name=<player>`; `intake` requires it. `/join
<pregen> as <name>`, `/sheet [name]`, `/party`. The REPL accepts
`Kara: I open the door` and the bot will map Discord authors the same way.
Out-of-turn actions in combat are rejected by `intake` with "It is Dorn's
turn" — no model call. Acceptance: two players alternate through the goblin
fight in the REPL; a third player's action during Kara's turn is refused
without reaching the model; the checkpoint restores the whole party.

**PR-22 — Discord bot (L)**
`feat/discord-bot`. `bot.py` with `discord.py`:
- `thread_id = channel id`; one `asyncio.Lock` per channel so turns serialise;
  `AsyncSqliteSaver`; `graph.astream`.
- Slash commands mirror the REPL commands; free text in a campaign channel is
  play; the author's display name is the player name.
- Streaming by editing one message at most every ~1 s (Discord rate limits);
  final text chunked at 2,000 characters; roll results as a small embed.
- Config via env: `DISCORD_TOKEN`, `DND_CAMPAIGN_CHANNELS` (allow-list).
- `docs/DISCORD.md`: creating the application, intents, inviting the bot,
  running it against the M4 over Tailscale.
Acceptance: two accounts play the goblin fight in a test server; the bot
survives a restart mid-encounter and resumes from the checkpoint; the REPL
still works (shared graph, two front ends).

**PR-23 — Session flow (S)**
`feat/sessions`. `/campaign new <premise>` seeds an opening scene; `/campaign
recap` narrates from `summary`; `/end` writes a session summary. A small
adventure premise library (three one-shots as data) so a first session needs
no prep. Acceptance: `/campaign new` then `/join` then play works cold from a
fresh database.

### Phase 4 — Spells and showcase

**PR-24 — Spellcasting v1 (L, optional)**
`feat/spells`. Slots per class/level from `Levels.json`, `cast(spell, target?)`
tool consuming a slot, attack-roll and save-based spells resolved by the
engine from `Spells.json` (damage dice, save ability), everything else
adjudicated by the DM with `lookup_rules`. Only after PR-20 shows the tool
loop is reliable — this is where argument complexity jumps.

**PR-25 — Showcase (S)**
`docs/showcase`. README rewrite around the Discord game, an annotated
transcript of a real session, the eval table, a graph diagram, and a "how
it stays honest" section (the three rules in §4). Refresh `KNOWN_ISSUES.md`
ledger.

---

## 6. Risks and how the plan handles them

| Risk | Handling |
|---|---|
| A tool-using turn on the Intel CPU is 2–4 generations (~40–80 s). | Engine does all mechanics with zero model calls; step cap; scene sheet keeps prompt-eval small; rolls never touch a model. The M4 is the intended host; the Intel path must stay *correct*, not fast. |
| A 7–8B model picks the wrong tool or invents an argument. | Every tool validates and answers in words; the model gets to try again inside the cap; PR-20 measures this before a model is chosen; the eval regexes catch invented numbers in prose. |
| Discord rate limits break streaming. | Edit-throttled streaming, chunked finals, a "thinking…" placeholder immediately. |
| Scope creep into a full 5e implementation. | v1 is martial combat, checks, saves, HP, conditions, rests. Spells are PR-24 and gated on PR-20. No multiclassing, no feats, no crafting. |
| Two front ends drift (REPL vs bot). | Both call the same compiled graph and the same command parser; the REPL is the test harness, not a product. |
| `qwen3` thinking mode floods CPU time. | Always `think=False` for the DM; measured in PR-20. |

## 7. What is explicitly not in this plan

- A web client or maps. Discord text is the product; a web UI is a later
  front end over the same graph.
- LLM-driven NPC agents with their own memory. The DM voices NPCs.
- Non-SRD content or the PDF corpus. `--source rulebooks` keeps working but
  nothing new depends on it.
- Cloud models. Local via Ollama, full stop.
- Voice, images, or a hosted service.

## 8. Reading order for a new session

1. `CLAUDE.md` — conventions, how to run.
2. This file — where we are going.
3. `docs/SPECS.md` — how the refactor got here, PR by PR.
4. `docs/KNOWN_ISSUES.md` — what is still open (#14, #16 close in Phase 1).
