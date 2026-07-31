# Agents

Four agents, all subclassing `BaseAgent`. Each registers itself as a graph node under
its `agent_type` string, which doubles as the routing token and the `agent` field in
the JSONL logs.

| `agent_type` | Class | File | Returns | Status |
|---|---|---|---|---|
| `supervisor` | `GameSupervisor` | `src/agents/supervisor.py` | `Command(goto=<agent>\|END)` | Works |
| `researcher` | `ResearcherAgent` | `src/agents/researcher.py` | `Command(goto="__end__")` | Works |
| `dice_roller` | `DiceRollerAgent` | `src/agents/dice_roller.py` | `Command(goto="supervisor")` | Works |
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

**Prompt** — `SUPERVISOR_PROMPT`: a four-line routing table (`dungeon_master` for
narrative, `researcher` for rules/lore, `dice_roller` for dice, `FINISH` when done)
ending with *"Return only the agent name."*

**Mechanism** — prepends `SUPERVISOR_PROMPT` as a `SystemMessage` to the existing
`BaseMessage` history (roles preserved, rather than flattening every turn to
`role="user"` as it used to), calls the LLM, then parses the response by
substring in fixed order:

```python
if   "dice_roller"    in text: goto = "dice_roller"
elif "dungeon_master" in text: goto = "dungeon_master"
elif "researcher"     in text: goto = "researcher"
elif "finish"         in text: goto = "FINISH"
else:                          goto = "researcher"
```

Any exception during the call is caught and also routes to `researcher`, with the
error recorded in the log metadata. `"FINISH"` is then mapped to LangGraph's `END`.

**Declared destinations** — `Command[Literal[*AGENT_TYPES, "__end__"]]` where
`AGENT_TYPES = ["dungeon_master", "researcher", "dice_roller"]`. The `Literal[*...]`
unpacking is why this module needs Python 3.11+.

**Things to know before changing it:**

- Substring matching means an LLM reply like *"not the dice_roller — use researcher"*
  routes to `dice_roller`. A local Llama 3.2 frequently produces prose around the
  answer.
- `researcher` is the fallback for *every* failure mode, which is why unroutable input
  ends up in RAG.
- The whole message history goes into every routing call, including prior assistant
  outputs. After a dice roll the newest message is the roll result, and the supervisor
  routes on *that* rather than on the original request.
- `class State(MessagesState)` and `class Router(TypedDict)` are declared at module
  scope and unused — `Router` looks like an abandoned attempt at structured output
  routing, which is the right fix (see `docs/REFACTOR_NOTES.md`).

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
Most of it disappears with a provider that supports structured outputs or tool calling.

**Roll step** (`_execute_dice_roll`) — for advantage/disadvantage, extracts the base
`NdM`, rolls twice via `DiceRoller.roll_multiple`, takes max/min, adds the modifier,
and formats both roll details. Otherwise rolls once, adds the modifier, and appends
per-die detail. Returns a Markdown string with a 🎲 prefix and a bolded total.

**Return** — `Command(goto="supervisor")` with one new `AIMessage`. That return
edge is what produces the extra supervisor hop described in
`docs/ARCHITECTURE.md`, and at ~5.7 s per routing call it is the most expensive
thing in the graph. PR-04 lets this node terminate directly instead.

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
