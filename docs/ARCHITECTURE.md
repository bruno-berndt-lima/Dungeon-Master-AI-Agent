# Architecture

## Shape of the system

```
                    main.py  (REPL)  /  bot.py (Discord, PR-22)
                       │  {messages: [HumanMessage], current_task}
                       ▼
          ┌────────────────────────────┐
          │  StateGraph(GameState)     │
          │  entry point: intake       │
          └────────────┬───────────────┘
                       ▼
                  ┌─────────┐   /rules q      ┌────────────┐
                  │ intake  │────────────────▶│ researcher │──▶ END
                  │ (code)  │   /roll /join   └────────────┘
                  └────┬────┘   /party, dice ─────────────────▶ END
                       │ play
                       ▼
              ┌────────────────┐  tool calls   ┌─────────┐
              │ dungeon_master │──────────────▶│  tools  │
              │  (model+tools) │◀──────────────│ (engine)│
              └───────┬────────┘  ToolMessages └─────────┘
                      │ narration                 ≤ 4 round trips per turn
                      ▼
              ┌────────────────┐
              │     memory     │  folds old messages into the journal
              └───────┬────────┘  when the transcript passes 24 (one internal call)
                      ▼
                     END
```

There are **no `add_edge` calls** and **no model-driven routing**. `intake`
is a function that returns `Command(goto=...)`; the DM and `tools` hand the
turn back and forth by `Command` until the model narrates. LangGraph derives
the legal destinations from each node's return annotation.

## A turn, end to end

1. **`main.py`** opens the SQLite checkpointer, compiles the graph against it,
   and picks a `thread_id` (`--thread`, or a fresh one).

2. User input becomes a `HumanMessage`. The first turn of a new thread seeds
   the default state; every later turn passes only `{"messages": [...],
   "current_task": ...}` and the checkpointer supplies the rest
   (`src/graph/campaigns.py`).

3. `game_graph.stream(turn, config=config, stream_mode="messages")` enters
   **`intake`** (`src/graph/intake.py`). In code, no model:
   - `/rules <q>` → `researcher`, with the bare question in `current_task`.
   - `/roll [n]` with a roll pending → `resolve_check` against the sheet, the
     result posted as an `intake` message, then → `dungeon_master` to narrate
     the outcome. With nothing pending, `/roll 2d6+3` rolls dice → END.
   - `/join`, `/party`, `/help`, and a bare dice request (`roll 1d20+5`) → END.
   - Anything else is play → `dungeon_master`, `tool_steps` reset to 0.

4. **`dungeon_master`** (`DungeonMaster.process_task`) builds the prompt —
   `DUNGEON_MASTER_PROMPT` + *the table right now* (`get_scene`) + the last
   six narrative messages + this turn's tool exchange — and calls the model
   with the tools bound. If the reply carries tool calls → `tools`; otherwise
   it is the narration and the turn ends. While a roll is pending, or once
   `MAX_TOOL_STEPS` (4) is spent, the model is called *without* tools and
   told to narrate what it has. See `docs/DM.md`.

5. **`tools`** (`DungeonMaster.run_tools`) runs each call through
   `run_tool` (`docs/TOOLS.md`) — which never raises — writes the `party` /
   `encounter` / `pending` deltas, and returns a `ToolMessage` per call.
   Back to step 4.

6. **`memory`** (`Memory.process_task`) runs after every narration. Past
   `MAX_MESSAGES` it folds the oldest messages into `summary` with one
   internal model call and removes them (`RemoveMessage`); otherwise it does
   nothing. See `docs/DM.md`, "Memory".

7. **`researcher`** — RAG: `question → scored retrieval → (rewrite + retry on
   a miss) → labelled passages → prompt → model → answer + sources`. Streams,
   returns `Command(goto="__end__")` with one new `AIMessage`.

8. Every model call and every tool call writes a line to
   `logs/llm_interactions/llm_log_<YYYY-MM-DD>.jsonl` with `stage` = `plan`,
   `tool`, `narrate`, or `fold`.

9. Back in `main.py`, streamed prose is not reprinted; other new messages are
   rendered; tool traffic is hidden unless `DND_SHOW_TOOLS=1`.

## Streaming

`main.py` consumes `stream_mode="messages"`, which yields `(chunk, metadata)` as
tokens are produced anywhere in the graph. No agent contains streaming code: a
plain `llm.invoke()` inside a node is routed through LangChain's streaming path
whenever a consumer is listening, so the tokens surface on their own.

Three filters make the stream readable, and each corresponds to a bug found while
building it:

- **`isinstance(chunk, AIMessageChunk)`** — the mode emits both per-token chunks
  *and* the finished `AIMessage` a node writes to state. Without this every
  narration prints twice.
- **`langgraph_node in STREAMING_NODES`** — the supervisor's routing call and the
  dice parse emit tokens too. Neither is for the player.
- **`"internal" not in tags`** — a single node can make several calls. The
  researcher rewrites a query before answering; both carry the same node name,
  so the internal one is tagged at the call site.
- **no `tool_call_chunks`** — the DM's tool-choosing reply streams too, as
  chunks of a JSON tool call with no content. They are not prose.

Measured on the target machine: first token ~3.6 s, against ~40 s to wait for a
finished narration. The researcher streams too since PR-08.

One wrinkle the researcher introduced: a node can add to its answer *after* the
model stops — `append_sources` lists the passages the answer came from. The
streamed text and the stored message therefore differ, so `main.py` prints the
unstreamed tail rather than skipping the message entirely. Without that the
citations never reached the player.

**Every worker terminates the turn** — the DM after its narration, the
researcher after its answer, `intake` directly for commands. Nothing routes
back into a router, because there is no router: PR-04 removed the
`dice_roller → supervisor` edge that once cost ~40 s of unwanted generation
per dice roll, and PR-18 removed the supervisor itself.

## State

`src/graph/game_state.py` declares `GameState(TypedDict)` with eight keys:

| Key | Type | Holds |
|---|---|---|
| `messages` | `Annotated[Sequence[BaseMessage], add_messages]` | full conversation |
| `current_task` | `str` | latest user input |
| `tool_steps` | `int` | tool round trips so far this turn; capped at 4 (PR-18) |
| `party` | `Dict[str, dict]` | character name → `Character` as JSON (PR-16) |
| `encounter` | `dict \| None` | the `Encounter` as JSON while a fight is on (PR-16) |
| `pending` | `dict \| None` | a `PendingCheck` as JSON while the DM waits on a roll (PR-16) |
| `summary` | `str` | the campaign journal, kept by `memory` (PR-19) |
| `last_response` | `str` | latest agent output |

The engine's models are stored as **plain JSON dicts**, never as pydantic
instances: LangGraph's checkpointer can serialise pydantic today but warns that
unregistered types will be refused later, and a registration list of every
engine type would rot. `get_party` / `put_party`, `get_encounter` /
`put_encounter`, `get_pending` / `put_pending` convert at the boundary.
`players`, `npcs`, `current_speaker`, `turn_order` and `requires_player_input`
— declared in the first commit and never written — went with `src/actors/`.

Two contracts to know before writing a node:

- **`messages` has a reducer; everything else replaces.** A node returns only the
  messages it produced and `add_messages` appends them. It merges on **message
  id**, not position — messages read out of state already carry ids, so
  re-returning them is deduped rather than duplicated. The real hazard is
  *rebuilding* message objects from scratch, which drops their ids and does
  duplicate. Both behaviors are pinned in `tests/test_state_contract.py`.
- **Routing lives only in `Command(goto=...)`.** There is no `next_agent` field.
  It used to mirror the routing decision, and because `ResearcherAgent` set it to
  `"FINISH"` on every successful answer, `main.py` exited the REPL after every
  rules question. Removed in PR-03.

`TypedDict` is not enforced at runtime, so a node can still write a key that
isn't declared — `tests/test_graph_smoke.py` guards the field set.

## Persistence

`create_game_graph(checkpointer=...)` accepts an optional
`langgraph.checkpoint.sqlite.SqliteSaver`; `create_sqlite_checkpointer()` builds
one over a long-lived connection. With a checkpointer attached, every `invoke`
needs `config={"configurable": {"thread_id": ...}}`, and each turn passes only
the new message — prior history is restored from the checkpoint.

**A campaign is a thread** (`src/graph/campaigns.py`, PR-13). `main.py` starts
a new one under a short id, or resumes one with `--thread <id>`; `--list` shows
every thread in the database with its turn count, location, and last input,
read through the checkpointer's own `list()` so the same code serves the async
saver the Discord bot will use. On resume the REPL prints a recap — location and
the DM's last line — from state, with no model call.

One rule the resume path depends on: **the default state is seeded only on a
thread's very first turn.** Every field but `messages` replaces on write, so
merging `create_default_game_state()` into the first turn of a *resumed*
session would overwrite the stored `game_state` with `{}` and the campaign
would forget where the party is. `seed_turn` decides by whether the thread has
any state yet; `tests/test_campaigns.py` pins it.

## Module responsibilities

**`src/agents/base_agent.py`** — `BaseAgent(ABC)` supplies three things to subclasses:
`initialize_agent(state)` (registers a stub entry under `state["game_state"]`, never
called by any subclass), `_log_interaction(...)`, and `_get_latest_message(state)`,
which tolerates both dict-shaped and `BaseMessage`-shaped history and falls back to
`current_task`. Abstract methods: `process_task`, `get_definition`.

**`src/models/llm.py`** — the single provider boundary. `create_llm(agent_type)`
resolves a model per role from `AGENT_MODELS` (`qwen2.5:7b` for both `researcher` and
`dungeon_master`), overridable by
`DND_MODEL_<AGENT_TYPE>` or `DND_MODEL_DEFAULT`, against the host in `OLLAMA_HOST`.
It returns an `OllamaChat` — a `ChatOllama` subclass that translates the two
failures this project hits constantly, a dead daemon and an unpulled model, into
messages that name the host and the `ollama pull` command. Construction makes no
network call, so the graph (and the test suite) build offline. Every agent
instantiates its own client in `__init__`.

**`src/prompts/prompts.py`** — three constants: `DUNGEON_MASTER_PROMPT` (the
tool contract and the voice), `NARRATE_ONLY_NOTE` (appended when tools are
withheld), `RESEARCHER_PROMPT`.

**`src/utils/dice.py`** — the cleanest module in the repo. `parse_dice_string`
splits on signed terms and yields `(quantity, sides)` tuples, skipping flat
modifiers (the caller adds those) and **raising** on anything the return type
cannot represent: a subtracted dice term, a zero quantity or die size, junk.
`roll_single_type` and `roll_multiple` return `DiceRoll` dataclasses carrying
`dice_type`, `results`, `total`. No LLM, no I/O, fully deterministic given
`random`.

**`src/utils/llm_logger.py`** — `LLMInteraction` dataclass plus `LLMLogger`, which
opens the day's JSONL file per write and appends. `get_recent_interactions(limit)`
reads back only the current day's file. Each agent constructs its own `LLMLogger`.

**`src/engine/`**, **`src/srd/`** — the deterministic 5e engine and the SRD as data; see `docs/ENGINE.md`.

**`src/data/`** — see `docs/RAG_PIPELINE.md`.

**`src/pipelines/`** — `create_retrieval_grader(llm)`, `create_question_rewriter(llm)`,
`create_rag_chain(llm)`. Together these are the standard corrective-RAG loop
(retrieve → grade relevance → rewrite query on failure → generate). None are imported
by any other module. `generator.py` also pulls `rlm/rag-prompt` from LangChain Hub,
which requires network access at construction time.

## Observability

The JSONL logs under `logs/llm_interactions/` are the only instrumentation, and they
are genuinely useful — 521 lines across four days, capturing the exact prompt sent
and reply received per agent, with a `stage` per line since PR-18 (`plan`, `tool`, `narrate`).
`llm_log_2025-03-31.jsonl` is kept as the evidence for KNOWN_ISSUES #6.
