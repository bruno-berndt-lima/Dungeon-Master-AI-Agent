# Agents

Four agents, all subclassing `BaseAgent`. Each registers itself as a graph node under
its `agent_type` string, which doubles as the routing token and the `agent` field in
the JSONL logs.

| `agent_type` | Class | File | Returns | Status |
|---|---|---|---|---|
| `supervisor` | `GameSupervisor` | `src/agents/supervisor.py` | `Command(goto=<agent>\|END)` | Works |
| `researcher` | `ResearcherAgent` | `src/agents/researcher.py` | `Command(goto="__end__")` | Works |
| `dice_roller` | `DiceRollerAgent` | `src/agents/dice_roller.py` | `Command(goto="__end__")` | Works |
| `dungeon_master` | `DungeonMaster` | `src/agents/dungeon_master.py` | `Command(goto="__end__")` | Works |

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
- Every destination now produces visible output. Between PR-04 and PR-06 a
  misroute to `dungeon_master` was *silent* — the stub returned `None` — which
  is what hid a routing bug during PR-04.
- `class State(MessagesState)` is deleted; `Router` is now the routing schema.

## `ResearcherAgent`

**Prompt** — `RESEARCHER_PROMPT`, rewritten in PR-08. It now tells the model that
each passage carries a source label, to cite the label rather than a remembered
page number, to say so when the passages do not answer the question, and to stop
after one short paragraph or five points.

**Retrieval** — `retrieve(question)` returns passages plus a metadata dict that
goes straight into the JSONL log:

```python
{"rag_used": True, "score": 0.09, "relevant": False, "rewritten": True,
 "rewritten_query": "Can a character use Sneak Attack after Hiding?",
 "retried_score": 0.302, "citations": ["Player's Handbook, p.175", ...]}
```

The **relevance gate is the retriever's own score**, not a model call. PR-08 was
specced to wire `create_retrieval_grader` here; measured, that cost **31.7 s per
query** and said "yes" every time, because it re-evaluates the same ~1,000-token
context the answer is about to be generated from. Over this index, on-topic
questions score 0.363–0.529 and off-topic ones -0.154–0.053, so
`RELEVANCE_THRESHOLD = 0.25` separates them for free. The threshold is specific
to this index and embedding model.

Below the threshold, the question is **rewritten once and retried** — never
looped. Player phrasing and rulebook phrasing sit far apart in embedding space:

| Question | Before | Rewritten to | After |
|---|---|---|---|
| "can my guy do the thing where he hides and stabs" | 0.09 | *Can your character use Sneak Attack after successfully hiding?* | 0.302 |
| "how do i make my dude tougher" | 0.041 | *How can you increase your character's Armor Class, Hit Points, and Constitution?* | 0.439 |

A retry that scores *worse* is discarded.

**Context** — `format_docs` labels every chunk with `[Book, p.N]`. It was defined
on the class from the first commit and never wired in; the chain interpolated the
`Document` objects instead, so their `repr` went into the prompt — chunk id,
`file_path`, `creationDate`, `trapped`, `format` and a dozen other PyMuPDF
fields. That was **1,653 tokens of prompt for four chunks**, most of it noise,
and it is why answers cited invented page numbers: `book` and `page_number` were
in there, buried in a Python dict repr. Clean formatting brings the same four
chunks to 1,062 tokens.

**Sources are appended deterministically.** The prompt asks for inline citation
and the model complies about half the time — it cited `Monster Manual, p.165`
unprompted for one question and nothing at all for the next. `append_sources`
lists the passages actually retrieved, so the player can always check the answer.
There is no reason to depend on the model for a fact this code already holds.

**Streaming** — a plain `invoke`, streamed by `main.py` through
`stream_mode="messages"` like the DM's narration. `num_predict=400` bounds the
answer; one unbounded answer measured 461 tokens and 181 s.

**Return** — always `Command(goto="__end__")`, on success and on error alike,
carrying one new `AIMessage`. This terminates the graph invocation, so the
researcher cannot hand back to the supervisor for a follow-up.

It used to also set `state["next_agent"] = "FINISH"`, which `main.py` read as a
signal to exit the REPL — so a single rules question ended the session. Both the
field and that behavior are gone (PR-03).

**If there is no index**, construction logs a warning and `retrieve` returns
nothing; the agent answers from the model alone and `metadata.rag_used` is
`False`. Check that field when an answer looks ungrounded.

## `DiceRollerAgent`

The only agent that combines an LLM with deterministic code, and the design is sound:
**the LLM parses, `DiceRoller` rolls.**

**Prompt** — `DICE_ROLLER_PROMPT` describes rolling behavior, but it is only ever
returned by `get_definition()`. The prompt that actually runs is
`DICE_PARSE_PROMPT`, now in `src/prompts/prompts.py` rather than inlined in the
class (PR-05, per the `CLAUDE.md` convention).

**Parse step** (`_parse_dice_request`) — **reads the request, and only falls back
to the model when the request names no dice.**

```
extract_dice_expression("roll 2d8 + 1d6 for damage")  -> ("2d8+1d6", 0)
extract_roll_flags("roll 1d20 with advantage")        -> (True, False, "")
```

A dice expression is a formal language, so a regex reads it exactly where a model
reads it approximately. Measured: `llama3.2:3b` invented `+1` on `2d8 + 1d6`,
`+5` on a plain advantage roll, `+2` on `roll for initiative` — see
KNOWN_ISSUES #27. The model is consulted only for requests like *"roll for
initiative"*, and even there the modifier comes from the player's words, never
from the model.

The common path costs **no LLM call at all**: 0.0 s, down from 4.6 s.

What PR-05 deleted, all of it defence against unreliable JSON from a 3B model:

- the markdown-fence regex and first-`{...}` extraction
- the `PQXYpqxy` check for template placeholders the model copied from the
  few-shot examples instead of substituting
- string-to-int coercion of `"+5"`
- description salvage by splitting on `for` / `to check` / `to see if`
- **the `("1d20", 0, False, False, "dice roll")` fallback** — the worst of them,
  because it answered a request nobody made with a confident number. An
  unreadable request now says so.

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

`DiceRollRequest(Dict[str, Any])` — class attributes with defaults on a `Dict`
subclass, never instantiated — is replaced by `DiceRequest(TypedDict)`, which is
the actual schema handed to `with_structured_output` (PR-05).

## `DungeonMaster`

Implemented in PR-06. It was `pass` until then — the reason the project exists,
and the one agent with no implementation.

**Prompt** — `DUNGEON_MASTER_PROMPT`, rewritten. The old one described the state
dict to the model (`game_state`, `active_agent`, `current_task`) and told it to
delegate to other agents, neither of which a narrator can act on. It now carries
voice (second person, present tense, sensory detail), table rules (never decide
what the player does, never roll dice — ask for the roll), and a **two-paragraph
limit**. Length is a latency control at 4.4 tok/s, not a style preference.

**Two LLM calls per turn:**

1. **Narration.** `temperature=0.8`, `num_predict=400` as a hard stop. Context is
   the last `CONTEXT_WINDOW` (4) messages, filtered to the player's own turns and
   the DM's own prior narration — a dice result or a rules citation is not part
   of the story, and feeding it back makes the DM narrate about mechanics.
2. **Scene extraction.** `with_structured_output(SceneUpdate)` over
   `SCENE_EXTRACTION_PROMPT`, pulling `location`, `items_gained`, `effects` out of
   the narration just produced. ~5.4 s, and it runs *after* the player has read
   the story. Tagged `internal` so its JSON never reaches the stream.

**World state** — the extraction result is folded into `state["game_state"]`:
`location` replaces, `inventory` and `effects` append without duplicates, and any
other key already there is preserved. The merge builds a new dict rather than
mutating — the aliasing bug in #8 was exactly this shape. On the next turn
`_scene_briefing` replays it as one line prepended to the system prompt, which is
what lets `CONTEXT_WINDOW` stay at 4 without losing continuity: the model gets
the *state* of the world instead of the transcript of it.

**Streaming** — `process_task` calls plain `invoke`. Under
`stream_mode="messages"` LangChain routes that through the streaming path anyway,
so no agent-side streaming code exists. Measured: first token ~3.6 s against
~40 s for a finished narration.

**Return** — `Command(goto="__end__")`, like every other worker since PR-04.

**Failure** — a model error or an empty narration still returns a `Command`
carrying the error as a message. Returning `None` *was* issue #1, so a test pins
it. Extraction failure is swallowed: the story is the product, world state is a
bonus.

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
