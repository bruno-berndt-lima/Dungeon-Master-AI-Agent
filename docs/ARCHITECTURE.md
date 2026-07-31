# Architecture

## Shape of the system

```
                    main.py  (REPL)
                       │  state dict
                       ▼
          ┌────────────────────────────┐
          │  StateGraph(GameState)     │
          │  entry point: supervisor   │
          └────────────┬───────────────┘
                       ▼
                 ┌───────────┐
                 │ supervisor│──── FINISH / unroutable ──▶ END
                 └─────┬─────┘
        ┌──────────────┼──────────────┐
   goto │         goto │         goto │
        ▼              ▼              ▼
  ┌───────────┐  ┌──────────┐  ┌─────────────┐
  │ researcher│  │dungeon_  │  │ dice_roller │
  │           │  │ master   │  │             │
  └─────┬─────┘  └────┬─────┘  └──────┬──────┘
        │ goto=       │ goto=         │ goto="__end__"
        │ "__end__"   │ "__end__"     │   uses src/utils/dice.py
        ▼             ▼               ▼
       END           END             END

  researcher ──▶ scored retrieval ──▶ [rewrite+retry on a miss] ──▶ ChatOllama
  supervisor ──▶ prefilter_route() ──▶ or ──▶ with_structured_output(Router)
  dungeon_master ──▶ narrate (streams) ──▶ extract scene ──▶ game_state
```

**Every worker terminates the turn.** Nothing routes back into the supervisor, so
a turn runs exactly one routing decision and one worker.

There are **no `add_edge` calls**. `create_game_graph()` registers four nodes and
sets `supervisor` as the entry point; all traversal is driven by each node returning
a `langgraph.types.Command(goto=..., update=...)`. LangGraph derives the legal
destinations from the return type annotation on `process_task`.

## A turn, end to end

1. **`main.py`** opens a SQLite checkpointer, compiles the graph against it, and
   mints a `thread_id` for the session.

2. User input becomes a `HumanMessage`. The first turn seeds the full default
   state; every later turn passes only `{"messages": [...], "current_task": ...}`
   and lets the checkpointer supply the rest.

3. `game_graph.stream(turn, config=config, stream_mode="messages")` enters the
   `supervisor` node. `main.py` streams rather than invokes so narration appears
   token by token — see "Streaming" below.

4. **`GameSupervisor.process_task`** routes in two stages.

   First `prefilter_route()` — a pure function, no model. If the request is
   unambiguously dice (`"roll 2d10 + 1d6"`, or bare notation like `"2d6+1d8"`)
   it returns `dice_roller` immediately. It is deliberately conservative: a
   question opener, or notation used descriptively (`"my sword does 2d6"`),
   returns `None` and falls through.

   Otherwise `SUPERVISOR_PROMPT` plus **the current request** — not the message
   tail — goes to `with_structured_output(Router, method="json_schema")`. `next`
   is a `Literal`, so the model cannot name a node that does not exist. It returns
   `Command(goto=<agent>, update={"active_agent": goto})`. Warm cost on the target
   machine: **~2.7 s** on `qwen2.5:7b` (`docs/KNOWN_ISSUES.md` #24, #25).

   There is **no fallback destination**. An unroutable turn ends with an explicit
   message; it does not become a `researcher` query.

5. The chosen node runs:

   - **`researcher`** — RAG. `question → scored retrieval → (rewrite + retry if
     the score misses) → labelled passages → prompt → LLM → answer + sources`.
     Streams, and returns `Command(goto="__end__")` with one new `AIMessage`.
   - **`dice_roller`** — reads the dice expression out of the request with a
     regex, rolls with the pure `DiceRoller` utility, returns
     `Command(goto="__end__")` with one new `AIMessage`. No LLM call unless the
     request names no dice. It used to return to the supervisor; see below.
   - **`dungeon_master`** — narrates the world's response, streaming as it goes,
     then makes a second structured call to lift durable facts (location,
     inventory, effects) into `game_state`. Returns `Command(goto="__end__")`.

6. Every agent call writes one line to `logs/llm_interactions/llm_log_<YYYY-MM-DD>.jsonl`
   with `timestamp`, `agent`, `query`, `response`, `metadata`.

7. Back in `main.py`, anything already printed live is skipped and the rest of
   this turn's messages are rendered whole. The loop continues until the user
   types `quit` or `exit` — no agent can end the session on their behalf.

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
- **`"internal" not in tags`** — a single node can make several calls. The DM
  narrates and then extracts world state as JSON; both carry the same node name,
  so the second is tagged at the call site. Without this the player sees raw JSON
  spliced onto the end of the story.

Measured on the target machine: first token ~3.6 s, against ~40 s to wait for a
finished narration. The researcher streams too since PR-08.

One wrinkle the researcher introduced: a node can add to its answer *after* the
model stops — `append_sources` lists the passages the answer came from. The
streamed text and the stored message therefore differ, so `main.py` prints the
unstreamed tail rather than skipping the message entirely. Without that the
citations never reached the player.

**The `dice_roller → supervisor` return edge is gone (PR-04).** It used to mean a
dice request took two supervisor turns, and the 2025-03-31 log shows what that
cost: after the roll, the supervisor saw the roll *result* as the newest message
and routed it to `researcher`, which spent ~40 s answering a question nobody
asked. Two changes close it — `dice_roller` terminates directly, and the
supervisor routes on `current_task` rather than the tail, so an agent's own
output can never become the thing being routed. A dice request now measures
**4.9 s** end to end instead of ~45 s.

## State

`src/graph/game_state.py` declares `GameState(TypedDict)` with ten keys:

| Key | Type | Holds |
|---|---|---|
| `messages` | `Annotated[Sequence[BaseMessage], add_messages]` | full conversation |
| `current_task` | `str` | latest user input |
| `active_agent` | `str` | set by the supervisor on each route |
| `game_state` | `Dict[str, Any]` | world state; stays a dict. Written by `dungeon_master`: `location`, `inventory`, `effects` |
| `players` / `npcs` | `Dict[str, Player/NPC]` | always `{}` — `src/actors/` is unused |
| `current_speaker` | `str` | never set |
| `turn_order` | `List[str]` | always `[]` |
| `last_response` | `str` | latest agent output |
| `requires_player_input` | `bool` | never read |

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
the new message — prior history is restored from the checkpoint. `main.py`
currently mints a fresh `thread_id` per run, so campaigns are not yet resumed
across sessions even though the storage supports it.

## Module responsibilities

**`src/agents/base_agent.py`** — `BaseAgent(ABC)` supplies three things to subclasses:
`initialize_agent(state)` (registers a stub entry under `state["game_state"]`, never
called by any subclass), `_log_interaction(...)`, and `_get_latest_message(state)`,
which tolerates both dict-shaped and `BaseMessage`-shaped history and falls back to
`current_task`. Abstract methods: `process_task`, `get_definition`.

**`src/models/llm.py`** — the single provider boundary. `create_llm(agent_type)`
resolves a model per role from `AGENT_MODELS` (`llama3.2:3b` for `supervisor` and
`dice_roller`, `qwen2.5:7b` for `researcher` and `dungeon_master`), overridable by
`DND_MODEL_<AGENT_TYPE>` or `DND_MODEL_DEFAULT`, against the host in `OLLAMA_HOST`.
It returns an `OllamaChat` — a `ChatOllama` subclass that translates the two
failures this project hits constantly, a dead daemon and an unpulled model, into
messages that name the host and the `ollama pull` command. Construction makes no
network call, so the graph (and the test suite) build offline. Every agent
instantiates its own client in `__init__`, so a four-agent graph opens four
clients; the daemon keeps both models resident, so this costs nothing here.

**`src/prompts/prompts.py`** — four constants: `DUNGEON_MASTER_PROMPT`,
`RESEARCHER_PROMPT`, `SUPERVISOR_PROMPT`, `DICE_ROLLER_PROMPT`. The supervisor prompt
is a strict "return only the agent name" instruction; the dice roller prompt describes
rolling behavior the agent doesn't actually delegate to the LLM (the LLM only parses;
`DiceRoller` rolls).

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

**`src/actors/`** — `Actor` ABC (`id`, `name`, `type`, `description`, `can_act()`,
`process_message()`), plus `Player` (always can act, `process_message` returns `None`)
and `NPC` (`NPCStats` dataclass, personality string, `process_message` is a TODO).
Imported by `game_state.py` for type hints; never instantiated anywhere.

**`src/data/`** — see `docs/RAG_PIPELINE.md`.

**`src/pipelines/`** — `create_retrieval_grader(llm)`, `create_question_rewriter(llm)`,
`create_rag_chain(llm)`. Together these are the standard corrective-RAG loop
(retrieve → grade relevance → rewrite query on failure → generate). None are imported
by any other module. `generator.py` also pulls `rlm/rag-prompt` from LangChain Hub,
which requires network access at construction time.

## Observability

The JSONL logs under `logs/llm_interactions/` are the only instrumentation, and they
are genuinely useful — 521 lines across four days, capturing the exact prompt sent
and reply received per agent, including the supervisor's routing decisions. When
debugging routing, read `llm_log_2025-03-31.jsonl`: it shows the full
supervisor → dice_roller → supervisor → researcher sequence for a single `roll 2d10 + 1d6`.
