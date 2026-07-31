# Agents

Four agents, all subclassing `BaseAgent`. Each registers itself as a graph node under
its `agent_type` string, which doubles as the routing token and the `agent` field in
the JSONL logs.

| `agent_type` | Class | File | Returns | Status |
|---|---|---|---|---|
| `supervisor` | `GameSupervisor` | `src/agents/supervisor.py` | `Command(goto=<agent>\|END)` | Works |
| `researcher` | `ResearcherAgent` | `src/agents/researcher.py` | `Command(goto="__end__")` | Works |
| `dice_roller` | `DiceRollerAgent` | `src/agents/dice_roller.py` | `Command(goto="__end__")` | Works |
| `dungeon_master` | `DungeonMaster` | `src/agents/dungeon_master.py` | `None` | **Unimplemented** |

## The `BaseAgent` contract

```python
class BaseAgent(ABC):
    def __init__(self, agent_type: str)          # sets self.agent_type, self.logger
    def initialize_agent(self, state) -> GameState   # registers a stub in state["game_state"]
    @abstractmethod
    def process_task(self, state: GameState) -> GameState
    @abstractmethod
    def get_definition(self) -> str
    def _log_interaction(self, query, response, metadata=None)
    def _get_latest_message(self, state) -> str
```

Two notes on the contract as written:

- `process_task` is annotated `-> GameState`, but every concrete implementation returns
  a `Command`. The subclass annotations (`-> Command[Literal[...]]`) are the ones
  LangGraph reads; the base annotation is vestigial.
- `initialize_agent` is defined but never called by any subclass, so
  `state["game_state"]` never receives per-agent entries.

`_get_latest_message` is the piece worth reusing: it reads the last message's
`content`, falling back to `current_task` when there is no history yet. Its old
dict branch is gone — the `add_messages` reducer coerces everything to
`BaseMessage`, so a plain dict can no longer reach an agent.

## `GameSupervisor`

**Prompt** — `SUPERVISOR_PROMPT`: a routing table with worked examples per
destination (`dice_roller`, `researcher`, `dungeon_master`, `FINISH`), closing on
*"Decide on intent, not on keywords."* It no longer says "return only the agent
name" — the schema enforces the shape, so the prompt only has to carry meaning.

**Mechanism** — two stages, since PR-04.

1. **`prefilter_route(request)`** — pure, no model. Returns `dice_roller` for
   bare notation (`"2d6+1d8"`) or notation with a roll verb (`"roll 2d10 for
   damage"`); returns `None` for anything else, including notation inside a
   question (`"what does d20 mean"`) or used descriptively (`"my sword does 2d6
   slashing"`). Conservative by design — a wrong fast answer is worse than a slow
   right one.
2. **Structured output** — `SUPERVISOR_PROMPT` plus the current request go to
   `with_structured_output(Router, method="json_schema")`. `Router["next"]` is a
   `Literal`, so constrained decoding makes an invalid destination impossible.

There is **no fallback destination.** A schema violation, an unknown value, or an
exception ends the turn with an explicit message and `router` recorded in the log
metadata. The old behavior — every failure silently becoming a `researcher`
query — produced confident RAG answers to questions nobody asked.

`"FINISH"` maps to `END` and emits a fixed closing line. Ending a turn with no
message at all is indistinguishable from a hang.

**Declared destinations** — `Command[Literal[*AGENT_TYPES, "__end__"]]` where
`AGENT_TYPES = ["dungeon_master", "researcher", "dice_roller"]`. The `Literal[*...]`
unpacking is why this module needs Python 3.11+.

**Things to know before changing it:**

- **It routes on `current_task`, not on message history.** That is what closes
  KNOWN_ISSUES #6: the tail of the conversation is whatever an agent last said,
  so routing on it meant routing on the answer instead of the question. The cost
  is that follow-ups carrying no standalone intent ("do that again") have no
  context to disambiguate. It also keeps the routing prompt constant-size, so
  latency does not grow across a campaign — pinned by a test.
- **Model choice is about accuracy, not speed.** `llama3.2:3b` scored 7/12 on a
  routing set where `qwen2.5:7b` scored 12/12, for ~1 s more per turn. See
  KNOWN_ISSUES #25 before moving it back.
- **A misroute to `dungeon_master` is currently silent** — that node is still a
  stub returning `None`, so the turn produces no output at all. Until PR-06,
  silence in the REPL means "routed to the DM", not "crashed".
- `class State(MessagesState)` is deleted; `Router` is now the routing schema.

## `ResearcherAgent`

**Prompt** — `RESEARCHER_PROMPT`: a D&D 5e knowledge assistant, told to answer from
the official rulebooks, cite page references where possible, format in Markdown, and
not to invent homebrew.

**Mechanism** — builds an LCEL chain in `__init__`:

```python
{"context": self.retriever, "question": RunnablePassthrough()}
    | ChatPromptTemplate.from_messages([
          ("system", RESEARCHER_PROMPT + "\n\nRelevant context:\n{context}"),
          ("user", "{question}")])
    | self.llm
    | StrOutputParser()
```

The whole construction is wrapped in `try/except`; if the vector store can't be
opened, it prints a warning and sets `self.rag_chain = None`, and `process_task`
silently falls back to a bare LLM call with no retrieval. The `metadata.rag_used`
field in the log records which path ran — check it when answers look ungrounded.

**Return** — always `Command(goto="__end__")`, on success and on error alike,
carrying one new `AIMessage`. This terminates the graph invocation, so the
researcher cannot hand back to the supervisor for a follow-up.

It used to also set `state["next_agent"] = "FINISH"`, which `main.py` read as a
signal to exit the REPL — so a single rules question ended the session. Both the
field and that behavior are gone (PR-03). The return annotation, which claimed
`Command[Literal["supervisor"]]` while returning `__end__`, now matches.

`format_docs` is defined on the class but never used; the retriever's `Document`
list is interpolated into the prompt directly. PR-08 resolves it.

**Default retriever settings** — `vectorstore.as_retriever()` with no arguments:
similarity search, `k=4`. Tuning `k` and switching to MMR are the cheapest quality
levers available.

## `DiceRollerAgent`

The only agent that combines an LLM with deterministic code, and the design is sound:
**the LLM parses, `DiceRoller` rolls.**

**Prompt** — `DICE_ROLLER_PROMPT` describes rolling behavior, but it is only ever
returned by `get_definition()`. The prompt that actually runs is an inline few-shot
JSON-extraction prompt inside `_parse_dice_request`, with three worked examples.

**Parse step** (`_parse_dice_request`) — asks for a JSON object with `dice_notation`,
`modifier`, `has_advantage`, `has_disadvantage`, `description`, then defends heavily
against the model:

- strips ` ```json ` fences via regex, extracts the first `{...}` block
- if `dice_notation` contains any of `PQXYpqxy`, treats it as an unsubstituted template
  placeholder and re-extracts the notation from the raw message with a regex
- coerces a string `modifier` like `"+5"` to an int
- if `description` is empty or literally `"roll"`, splits the message on `for` /
  `to check` / `to see if` to recover intent
- on any exception, returns `("1d20", 0, False, False, "dice roll")`

That defensive stack is a direct artifact of prompting a small local model for JSON.
PR-05 replaces it with `with_structured_output` over a strict schema — the same
mechanism PR-04 used for routing, which is available locally on
`langchain-ollama` 1.1.0.

**Roll step** (`_execute_dice_roll`) — for advantage/disadvantage, extracts the base
`NdM`, rolls twice via `DiceRoller.roll_multiple`, takes max/min, adds the modifier,
and formats both roll details. Otherwise rolls once, adds the modifier, and appends
per-die detail. Returns a Markdown string with a 🎲 prefix and a bolded total.

**Return** — `Command(goto="__end__")` with one new `AIMessage`. It returned to
the supervisor until PR-04. The hop itself was cheap (~0.65 s); what made it the
most expensive thing in the graph was the ~40 s researcher answer it went on to
trigger, because the supervisor then routed on the roll *result*. A roll is a
complete answer, so the node now terminates. A dice request measures 4.9 s end to
end, down from ~45 s.

It previously did `dict(state)` — a shallow copy — and then appended to
`updated_state["messages"]`, writing through to the graph's own list. Fixed in
PR-03.

`DiceRollRequest(Dict[str, Any])` at the top of the file declares class attributes with
defaults on a `Dict` subclass — it is not a usable structure and is never instantiated.
It should be a `pydantic.BaseModel` or a `TypedDict`.

## `DungeonMaster`

Fully stubbed:

```python
def process_task(self, state: GameState) -> Command[Literal["supervisor"]]:
    """Processes a game-related task."""
    pass
```

The class is constructed in `create_game_graph()` and registered as a node, so the
supervisor can route to it — and when it does, the node returns `None` instead of a
`Command`. `DUNGEON_MASTER_PROMPT` exists and is assigned to `self.system_prompt`,
`create_llm()` is instantiated, and nothing consumes either.

This is the single largest gap: the narrative agent is the reason the project exists,
and it is the one agent with no implementation. Everything it needs to be written
against — prompt, LLM factory, state schema, logging, the `Command` return pattern —
is already in place; see `docs/REFACTOR_NOTES.md` for the shape it should take.

## Adding an agent

1. Add the module under `src/agents/`, subclassing `BaseAgent`.
2. Add its prompt as a constant in `src/prompts/prompts.py`.
3. Call `super().__init__("<agent_type>")` — this string is the node name, the routing
   token, and the log key, so keep it identical everywhere.
4. Annotate `process_task` with the destinations it can actually return.
5. Call `self._log_interaction(...)` on every LLM call.
6. Register the node in `create_game_graph()`.
7. Add the routing token to `AGENT_TYPES` in `supervisor.py` **and** to the parse chain
   in `process_task` **and** to `SUPERVISOR_PROMPT`. All three are separate places
   today — consolidating them into one registry is a worthwhile refactor.
