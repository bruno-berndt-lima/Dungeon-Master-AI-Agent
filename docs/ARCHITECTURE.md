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
        ┌────────│ supervisor│◀───────┐
        │        └─────┬─────┘        │
        │              │              │ Command(goto="supervisor")
   goto │         goto │         goto │
        ▼              ▼              │
  ┌───────────┐  ┌──────────┐  ┌──────┴──────┐
  │ researcher│  │dungeon_  │  │ dice_roller │
  │           │  │ master   │  │             │
  └─────┬─────┘  └──────────┘  └─────────────┘
        │ goto="__end__"           uses
        ▼                    src/utils/dice.py
       END

  researcher ──▶ Chroma retriever ──▶ prompt ──▶ ChatOllama ──▶ str
```

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

3. `game_graph.invoke(turn, config=config)` enters the `supervisor` node.

4. **`GameSupervisor.process_task`** prepends `SUPERVISOR_PROMPT` as a
   `SystemMessage` to the existing history and calls the LLM. The reply is parsed
   by **substring match** against `"dice_roller"`, `"dungeon_master"`,
   `"researcher"`, `"finish"` — first hit wins, in that order. Anything unmatched,
   an exception, or a `None` response defaults to `researcher`. It returns
   `Command(goto=<agent>, update={"active_agent": goto})`. Measured cost of this
   call on the target machine: **~5.7 s** (see `docs/KNOWN_ISSUES.md` #24).

5. The chosen node runs:

   - **`researcher`** — RAG. `latest_message → retriever → context → prompt → LLM → str`.
     Returns `Command(goto="__end__")` with one new `AIMessage`.
   - **`dice_roller`** — LLM-parses the request into structured fields, rolls with the
     pure `DiceRoller` utility, returns `Command(goto="supervisor")` with one new
     `AIMessage`.
   - **`dungeon_master`** — `process_task` is `pass`. Returns `None`. This node cannot
     currently execute successfully.

6. Every agent call writes one line to `logs/llm_interactions/llm_log_<YYYY-MM-DD>.jsonl`
   with `timestamp`, `agent`, `query`, `response`, `metadata`.

7. Back in `main.py`, only the messages added by this turn are printed. The loop
   continues until the user types `quit` or `exit` — no agent can end the session
   on their behalf.

The `dice_roller → supervisor` return edge means dice requests take at least two
supervisor turns. The 2025-03-31 log shows the practical consequence: after the roll,
the supervisor sees the roll result as the newest message and routes to `researcher`,
which then answers a question nobody asked. The routing prompt has no notion of
"the request is already satisfied." At ~5.7 s per routing call, that wrong turn is
also the single most expensive thing the graph does. PR-04 addresses both.

## State

`src/graph/game_state.py` declares `GameState(TypedDict)` with ten keys:

| Key | Type | Holds |
|---|---|---|
| `messages` | `Annotated[Sequence[BaseMessage], add_messages]` | full conversation |
| `current_task` | `str` | latest user input |
| `active_agent` | `str` | set by the supervisor on each route |
| `game_state` | `Dict[str, Any]` | world state; stays a dict |
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

**`src/models/llm.py`** — the single provider boundary. `create_llm(model_name="Llama3.2",
temperature=0)` returns a `ChatOllama`; `create_json_llm(...)` adds `format="json"`
and is never called. Every agent instantiates its own client in `__init__`, so a
four-agent graph opens four Ollama clients.

**`src/prompts/prompts.py`** — four constants: `DUNGEON_MASTER_PROMPT`,
`RESEARCHER_PROMPT`, `SUPERVISOR_PROMPT`, `DICE_ROLLER_PROMPT`. The supervisor prompt
is a strict "return only the agent name" instruction; the dice roller prompt describes
rolling behavior the agent doesn't actually delegate to the LLM (the LLM only parses;
`DiceRoller` rolls).

**`src/utils/dice.py`** — the cleanest module in the repo. `DiceRoller.parse_dice_string`
splits on `+` and yields `(quantity, sides)` tuples; `roll_single_type` and
`roll_multiple` return `DiceRoll` dataclasses carrying `dice_type`, `results`, `total`.
No LLM, no I/O, fully deterministic given `random`. Note it only handles `+` — a
negative modifier written as `2d6-1` is silently dropped by the parser (the agent
extracts modifiers separately via the LLM, which is why this mostly works in practice).

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
