# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

An AI Dungeon Master for D&D 5e: a LangGraph multi-agent system where a supervisor
routes player input to specialist agents (narrator, rules researcher, dice roller).
Rules retrieval is RAG over the three core 5e rulebooks, indexed in a local ChromaDB.
All inference currently runs locally through Ollama.

The project **runs**: all four agents are implemented, routing is schema-constrained,
and narration streams. Some modules are still scaffolding that is never called
(`src/pipelines/`, `src/actors/`). See
`docs/KNOWN_ISSUES.md` before assuming any given path executes.

## Running it

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
ollama serve &              # or launch Ollama.app
ollama pull llama3.2:3b     # note the tag: "Llama3.2" does NOT resolve
python main.py              # interactive REPL; type "quit" or "exit" to leave
```

**Use Python 3.12 specifically**, not 3.11 and not 3.13+. Two independent
constraints pin it:

- `src/agents/supervisor.py` uses `Literal[*ROUTING_OPTIONS]` (PEP 646), a syntax
  error before **3.11**.
- On Intel macOS the last torch release with an x86_64 wheel is **2.2.2**, whose
  newest interpreter tag is **cp312**. Above 3.12 there is no installable torch,
  and `sentence-transformers` needs it — so retrieval will not build.

`requirements.txt` also pins `transformers` and `numpy` for the same reason; the
comments there explain each. Both torch/numpy pins carry platform markers, so
they are inert off Intel macOS.

Requires a running Ollama daemon. The `chroma_db/` directory is committed, so
retrieval works out of the box — you do **not** need the source PDFs to run the app.
They are only required to re-index, and they are gitignored (`Documents/README.md`).
See "Rebuilding the index" below.

**Expect generation to be slow.** Ollama has no GPU path on Intel Macs, so this
is CPU-only: **11.4 tok/s** on `llama3.2:3b`, **5.3 tok/s** on `qwen2.5:7b` — a
200-token answer takes 18 s and 38 s respectively. Routing is not the problem
(~0.65 s warm); output tokens are. The first call to each model also pays a cold
load of 5–11 s. See `docs/KNOWN_ISSUES.md` #24.

## Layout

| Path | Role |
|---|---|
| `main.py` | REPL loop; builds state, streams the compiled graph token by token |
| `src/config.py` | Chroma dir, PDF paths, embedding model name |
| `src/graph/game_orchestrator.py` | Builds the `StateGraph`, registers agent nodes |
| `src/graph/game_state.py` | `GameState` TypedDict + default factory |
| `src/agents/` | `base_agent` (ABC), `supervisor`, `dungeon_master`, `researcher`, `dice_roller` |
| `src/actors/` | `Actor` ABC, `Player`, `NPC` — data models, not yet used by the graph |
| `src/data/` | `loader` (PDF), `processing` (chunking), `vectorstore` (Chroma) |
| `scripts/ingest.py` | Rebuilds `chroma_db/` from the PDFs; `--rebuild`, `--dry-run` |
| `src/pipelines/` | `grader`, `rewriter`, `generator` — corrective-RAG parts, currently unused |
| `src/models/llm.py` | `create_llm(agent_type)` — the single LLM factory; per-agent model map, env overrides |
| `src/prompts/prompts.py` | All system prompts, as module-level string constants |
| `src/utils/dice.py` | Pure dice notation parser + roller (no LLM) |
| `src/utils/llm_logger.py` | Appends every agent call to `logs/llm_interactions/*.jsonl` |
| `Documents/` | Where the three 5e PDFs go. **Gitignored** — supply your own; see `Documents/README.md` |
| `chroma_db/` | Persisted vector store, 4778 chunks, 384-dim (committed) |

## Conventions to follow

- **One LLM factory.** Every agent calls `create_llm(self.agent_type)` from
  `src/models/llm.py` — after `super().__init__(...)`, so `agent_type` is set. The
  model is chosen there, per role, from `AGENT_MODELS`; a new agent type not in
  that map falls back to `DEFAULT_MODEL`. Change models, host, or provider in that
  one file, never in an agent. Overrides without editing code:
  `DND_MODEL_<AGENT_TYPE>`, `DND_MODEL_DEFAULT`, `OLLAMA_HOST`.
- **Prompts live in `src/prompts/prompts.py`** as `UPPER_SNAKE` constants, imported
  by name. Don't inline system prompts in agent classes. (`DiceRollerAgent._parse_dice_request`
  currently violates this with an inline parse prompt.)
- **New agents subclass `BaseAgent`** and implement `process_task(state)` and
  `get_definition()`. `__init__` must call `super().__init__("<agent_type>")` — that
  string is the node name, the log `agent` field, and the routing token.
- **Every LLM call gets logged** via `self._log_interaction(query, response, metadata)`.
  Keep this when adding agents; the JSONL logs are the only observability here.
- **Routing is Command-based, not edge-based.** Nodes return
  `Command(goto=..., update={...})`. The return type annotation
  (`Command[Literal["supervisor"]]`) is what LangGraph reads to infer valid
  destinations — it must match what the method actually returns. There is no
  `next_agent` state field; `goto` is the only routing channel.
- **Return deltas, not whole state.** A node returns only the keys it changed.
  For `messages` that means only the messages it produced — the `add_messages`
  reducer appends them. Never `dict(state)` and mutate: that is a shallow copy,
  so you write through to the graph's own lists.
- **Messages are `BaseMessage`** (`HumanMessage` / `AIMessage`, with `name` set
  to the agent type). The reducer coerces anything else, so agents always read
  message objects. Use `BaseAgent._get_latest_message` rather than indexing.

## Rebuilding the index

```bash
python scripts/ingest.py              # build; refuses to touch an existing index
python scripts/ingest.py --rebuild    # replace it
python scripts/ingest.py --dry-run    # load and chunk without embedding
```

Needs the three PDFs in `Documents/` — gitignored, so supply your own
(`Documents/README.md`). The script names the missing files rather than failing
on a traceback. You do **not** need them to run the app; `chroma_db/` is
committed.

Under the hood: `load_documents` (PyMuPDF, tags `book` + `page_number`) →
`split_documents` (1000 chars, 200 overlap) → `build_vectorstore`.

`load_vectorstore()` reads and `build_vectorstore(docs, rebuild=...)` writes —
they used to be one function that silently ignored its `docs` argument whenever
`chroma_db/` existed. Building over an existing store is refused, because Chroma
appends and would duplicate every chunk. Embeddings are `all-MiniLM-L6-v2`
(384-dim); changing the embedding model invalidates the whole index.

## Testing

```bash
pytest                      # 29 passed, 1 xfailed
pytest -m "not integration" # unit tests only — no dependency stack needed
```

Three files:

- `tests/test_dice.py` — `src/utils/dice.py`, pure and offline. Includes a strict
  `xfail` pinning the `-` modifier bug (KNOWN_ISSUES #9); PR-05 flips it.
- `tests/test_state_contract.py` — the `add_messages` reducer and the SQLite
  checkpointer, exercised with a stand-in node so **no model daemon is needed**.
- `tests/test_graph_smoke.py` — the graph compiles and every agent node registers.

The last two are marked `integration`: they need the dependency stack and Python
3.12, but not a running model. Nothing in the suite calls a model — keep it that
way, so the gate stays fast and runnable offline.

## Further reading

- `SPECS.md` — **the execution contract for the refactor: one spec per PR**
- `docs/ARCHITECTURE.md` — how a turn flows through the system, module by module
- `docs/AGENTS.md` — per-agent contracts, prompts, and routing behavior
- `docs/RAG_PIPELINE.md` — retrieval, chunking, the index, and the unused CRAG parts
- `docs/KNOWN_ISSUES.md` — verified bugs and dead code, ranked
- `docs/REFACTOR_NOTES.md` — direction, measured local-model performance, and what's left
