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

1. **`main.py`** builds a `GameState` via `create_default_game_state()`, then
   `.update()`s it with a partly-overlapping set of keys (`current_task`, `messages`,
   `active_agent`, `game_state`, `players`, `npcs`, `turn_order`, `next_agent`).
   Note `game_state` is reset from `{}` to the string `"initialized"` here — see
   `docs/KNOWN_ISSUES.md`.

2. User input is appended to `state["messages"]` as `{"role": "user", "content": ...}`
   and mirrored into `state["current_task"]`.

3. `game_graph.invoke(state)` enters the `supervisor` node.

4. **`GameSupervisor.process_task`** short-circuits to `END` if `next`/`next_agent`
   is `"FINISH"`. Otherwise it flattens the whole message history into an OpenAI-style
   message list, prepends `SUPERVISOR_PROMPT`, and calls the LLM. The reply is parsed
   by **substring match** against `"dice_roller"`, `"dungeon_master"`, `"researcher"`,
   `"finish"` — first hit wins, in that order. Anything unmatched, an exception, or a
   `None` response defaults to `researcher`. It returns
   `Command(goto=<agent>, update=state|next_agent)`.

5. The chosen node runs:

   - **`researcher`** — RAG. `latest_message → retriever → context → prompt → LLM → str`.
     Appends an assistant message and returns `Command(goto="__end__")`.
   - **`dice_roller`** — LLM-parses the request into structured fields, rolls with the
     pure `DiceRoller` utility, appends a formatted result, returns
     `Command(goto="supervisor")`.
   - **`dungeon_master`** — `process_task` is `pass`. Returns `None`. This node cannot
     currently execute successfully.

6. Every agent call writes one line to `logs/llm_interactions/llm_log_<YYYY-MM-DD>.jsonl`
   with `timestamp`, `agent`, `query`, `response`, `metadata`.

7. Back in `main.py`, the returned state is reassigned, `current_task` cleared, and the
   loop prints every state key before prompting again.

The `dice_roller → supervisor` return edge means dice requests take at least two
supervisor turns. The 2025-03-31 log shows the practical consequence: after the roll,
the supervisor sees the roll result as the newest message and routes to `researcher`,
which then answers a question nobody asked. The routing prompt has no notion of
"the request is already satisfied."

## State

`src/graph/game_state.py` declares `GameState(TypedDict)` with eleven keys:

| Key | Declared type | Actually holds |
|---|---|---|
| `messages` | `Sequence[BaseMessage]` | list of plain dicts |
| `current_task` | `str` | latest user input |
| `active_agent` | `str` | set once, never updated |
| `game_state` | `dict` | `dict` from the factory, `str` after `main.py` |
| `players` / `npcs` | `Dict[str, Player/NPC]` | always `{}` |
| `current_speaker` | `str` | never set |
| `turn_order` | `List[str]` | always `[]` |
| `last_response` | `str` | never set |
| `requires_player_input` | `bool` | never read |
| `next_agent` | `str` | routing decision, mirrored by supervisor |

`TypedDict` is not enforced at runtime, so mismatches don't raise. Two consequences
worth knowing before refactoring:

- **No reducers.** Nothing is `Annotated[..., add_messages]`, so a node returning
  `messages` replaces the list rather than appending to it. Agents work around this
  by copying the incoming list and re-returning the whole thing.
- **Two parallel routing channels.** `Command(goto=...)` is what LangGraph actually
  follows; `state["next_agent"]` is a redundant mirror that only `main.py` and the
  supervisor's FINISH check read.

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
