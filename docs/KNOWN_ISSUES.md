# Known issues

Findings from a full read of every source file. Each is traceable to a specific line;
none are speculative. Ordered by impact.

## Blocking

### 1. `DungeonMaster.process_task` is a stub

`src/agents/dungeon_master.py:23-25` — the body is `pass`, so the node returns `None`
instead of a `Command`. The node is registered in the graph and the supervisor can
route to it, so any narrative input reaches a dead end. This is the project's core
agent.

### 2. `Literal[*ROUTING_OPTIONS]` requires Python 3.11+

`src/agents/supervisor.py:12` (and `:31`) use PEP 646 unpacking inside a subscript.
On the system Python here (3.9.6) the module fails at import, which takes the whole
graph with it. `requirements.txt` and `pyproject.toml` declare no `python_requires`.

### 3. Return-type annotations don't match returns

LangGraph infers a node's legal destinations from the `Command[Literal[...]]`
annotation:

- `ResearcherAgent.process_task` — annotated `Command[Literal["supervisor"]]`,
  returns `goto="__end__"` (`src/agents/researcher.py:57`, `:113`, `:145`)
- `DiceRollerAgent.process_task` — annotated `Command[Literal["supervisor"]]` and does
  return there, so this one is consistent
- `BaseAgent.process_task` — annotated `-> GameState`, but every implementation returns
  a `Command`

The researcher mismatch is the one that can surprise LangGraph at compile or run time.

## Correctness

### 4. `game_state` changes type between the factory and `main.py`

`create_default_game_state()` sets `game_state={}` (a dict).
`main.py:31-40` then overwrites it with the string `"initialized"`.
`BaseAgent.initialize_agent` does `if self.agent_type not in state["game_state"]`,
which on a string performs a substring check rather than a key lookup. The method is
never called today, so nothing breaks — but it will the moment an agent uses it.

### 5. Supervisor routes by substring match

`src/agents/supervisor.py:52-66`. The parse chain checks `"dice_roller" in text`,
then `"dungeon_master"`, then `"researcher"`, then `"finish"`. A reply such as
*"This isn't for the dice_roller, send it to the researcher"* routes to `dice_roller`.
Small local models routinely wrap the answer in prose. Every failure path
(no match, exception, `None` response) falls through to `researcher`.

### 6. No terminal condition after a dice roll

`dice_roller` returns to `supervisor`, which then re-reads the full history. The newest
message is now the roll result, so the supervisor routes on the *result* rather than
the original request. `logs/llm_interactions/llm_log_2025-03-31.jsonl` captures this
exactly: `roll 2d10 + 1d6` → `dice_roller` → `supervisor` → `researcher`, which then
answers a rules question nobody asked. `SUPERVISOR_PROMPT` has no concept of "already
satisfied."

### 7. `GameState` declares no reducers

Nothing in `src/graph/game_state.py` is `Annotated[..., add_messages]`. Returned values
replace rather than merge, which is why every agent copies the message list and returns
it whole. Any new node that returns a partial `messages` list will silently drop history.

### 8. `dice_roller` mutates shared caller state

`src/agents/dice_roller.py:52-59` — `updated_state = dict(state)` is a shallow copy, so
`updated_state["messages"].append(...)` mutates the caller's list. `ResearcherAgent`
does this correctly (`updated_messages = list(...)` first); the two should match.

### 9. `DiceRoller.parse_dice_string` only splits on `+`

`src/utils/dice.py:22` — `dice_str.split("+")`. A notation like `2d6-1` yields a part
of `2d6-1`, which then fails `int("6-1")`. In practice the agent extracts modifiers
separately via the LLM and passes clean notation, so this rarely surfaces — but the
utility is not safe to call directly with arbitrary notation.

### 10. `Chroma` imported twice

`src/data/vectorstore.py:3-4` imports `Chroma` from `langchain_community.vectorstores`
then immediately from `langchain_chroma`. The second wins. Remove the first.

### 11. `get_vectorstore` silently ignores its argument

`src/data/vectorstore.py:12-16` — when `chroma_db/` exists, `docs` is discarded and the
existing store is returned. There's no incremental-add path and no way to tell from the
call site whether documents were indexed. `ResearcherAgent` exploits this by passing
`[]`. Split into `load_vectorstore()` and `build_vectorstore(docs)`.

## Dead code

### 12. No ingestion entry point

`load_documents` and `split_documents` are complete and correct but imported nowhere.
The committed `chroma_db/` was built out-of-band and cannot be reproduced from the repo
without writing the script yourself.

### 13. The entire `src/pipelines/` package is unused

`create_retrieval_grader`, `create_question_rewriter`, `create_rag_chain` — no importers.
This is a complete corrective-RAG loop that was never wired in. `generator.py`
additionally requires network access at construction (`hub.pull("rlm/rag-prompt")`) and
defines a `format_docs` it never uses.

### 14. `src/actors/` is unused

`Player` and `NPC` are imported by `game_state.py` for type hints only. Neither is ever
instantiated; `state["players"]` and `state["npcs"]` are always `{}`.
`NPC.process_message` is an explicit `TODO`.

### 15. Unused declarations

- `supervisor.py` — `class Router(TypedDict)` and `class State(MessagesState)`, both
  module-scope, both unreferenced. `Router` looks like an abandoned structured-output
  routing attempt, which is the right approach.
- `dice_roller.py` — `class DiceRollRequest(Dict[str, Any])` declares class attributes
  with defaults on a `Dict` subclass; it isn't a usable structure and is never
  instantiated.
- `researcher.py` — `format_docs` defined but never wired into the chain.
- `models/llm.py` — `create_json_llm` never called.
- `base_agent.py` — `initialize_agent` never called.
- `requirements.txt` — `tavily-python` and `langchainhub` are listed; Tavily is never
  imported, and `langchainhub` is only reachable through the unused `generator.py`.

### 16. Six state keys are never written

`current_speaker`, `last_response`, `requires_player_input`, `turn_order`, `players`,
`npcs` — declared in `GameState`, never assigned outside the default factory.

## Tooling

### 17. `tests/` are not tests

Both files are print-driven scripts with no assertions, requiring a live Ollama daemon
and a populated Chroma index. `test_dice_roller.py` prints
`"Command result successfully returned"` unconditionally — it cannot fail.

### 18. `pyproject.toml` pytest config is in the wrong table

`[tool.pytest]` is ignored by pytest; the recognized table is
`[tool.pytest.ini_options]`. `testpaths` and `pythonpath` currently have no effect
(`conftest.py` is doing the path work instead).

### 19. `requirements.txt` pins nothing and duplicates `langchain`

Eleven unpinned packages, with `langchain` listed twice. LangGraph's `Command` API and
LangChain's package split have both moved since; an unpinned reinstall today will not
reproduce the environment this code was written against.

### 20. Large binaries committed

`Documents/` (~370 MB of PDFs) and `chroma_db/` are both in git. `.gitignore` has
`*.db` and `*.sqlite`, neither of which matches `chroma.sqlite3`. Beyond repo size,
the PDFs are commercial Wizards of the Coast rulebooks in a public repository — see
the licensing note in `docs/RAG_PIPELINE.md`.

### 21. `env_activation.txt` is Windows-only

Contains `.\venv\Scripts\activate`. Fine as a personal note; misleading on this macOS
checkout, where the command is `source venv/bin/activate`.

## Suggested order of attack

If the goal is a working system before a rewrite:

1. Pin Python ≥3.11 and pin dependency versions (#2, #19)
2. Implement `DungeonMaster.process_task` (#1)
3. Fix the researcher's return annotation (#3)
4. Replace substring routing with structured output (#5)
5. Add a terminal condition so `dice_roller` doesn't trigger a spurious research turn (#6)
6. Add `add_messages` reducers to `GameState` (#7)
7. Write `scripts/ingest.py` (#12)

Items 4 and 6 get substantially easier after the provider swap — see
`docs/REFACTOR_NOTES.md`.
