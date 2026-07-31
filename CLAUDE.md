# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

An AI Dungeon Master for D&D 5e: a LangGraph multi-agent system where a supervisor
routes player input to specialist agents (narrator, rules researcher, dice roller).
Rules retrieval is RAG over the three core 5e rulebooks, indexed in a local ChromaDB.
All inference currently runs locally through Ollama.

The project is a **working prototype, not a finished app**. Several modules are
scaffolding that is never called, and the graph does not fully wire up. See
`docs/KNOWN_ISSUES.md` before assuming any given path executes.

## Running it

```bash
python -m venv venv && source venv/bin/activate   # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
ollama pull llama3.2        # src/models/llm.py hardcodes model name "Llama3.2"
python main.py              # interactive REPL; type "quit" or "exit" to leave
```

Requires a running Ollama daemon. The `chroma_db/` directory is committed, so
retrieval works without a re-index — there is no ingestion script in the repo
(see "Rebuilding the index" below).

**Python version matters.** `src/agents/supervisor.py` uses `Literal[*ROUTING_OPTIONS]`,
which is PEP 646 syntax requiring **Python 3.11+**. The system Python here is 3.9.6,
so the module will fail to import on it. Use 3.11 or newer in the venv.

## Layout

| Path | Role |
|---|---|
| `main.py` | REPL loop; builds state, invokes the compiled graph |
| `src/config.py` | Chroma dir, PDF paths, embedding model name |
| `src/graph/game_orchestrator.py` | Builds the `StateGraph`, registers agent nodes |
| `src/graph/game_state.py` | `GameState` TypedDict + default factory |
| `src/agents/` | `base_agent` (ABC), `supervisor`, `dungeon_master`, `researcher`, `dice_roller` |
| `src/actors/` | `Actor` ABC, `Player`, `NPC` — data models, not yet used by the graph |
| `src/data/` | `loader` (PDF), `processing` (chunking), `vectorstore` (Chroma) |
| `src/pipelines/` | `grader`, `rewriter`, `generator` — corrective-RAG parts, currently unused |
| `src/models/llm.py` | `create_llm()` / `create_json_llm()` — the single LLM factory |
| `src/prompts/prompts.py` | All system prompts, as module-level string constants |
| `src/utils/dice.py` | Pure dice notation parser + roller (no LLM) |
| `src/utils/llm_logger.py` | Appends every agent call to `logs/llm_interactions/*.jsonl` |
| `Documents/` | The three 5e PDFs (~370 MB, committed) |
| `chroma_db/` | Persisted vector store, 4778 chunks, 384-dim (committed) |

## Conventions to follow

- **One LLM factory.** Every agent calls `create_llm()` from `src/models/llm.py`.
  Change the model/provider there, not in individual agents.
- **Prompts live in `src/prompts/prompts.py`** as `UPPER_SNAKE` constants, imported
  by name. Don't inline system prompts in agent classes. (`DiceRollerAgent._parse_dice_request`
  currently violates this with an inline parse prompt.)
- **New agents subclass `BaseAgent`** and implement `process_task(state)` and
  `get_definition()`. `__init__` must call `super().__init__("<agent_type>")` — that
  string is the node name, the log `agent` field, and the routing token.
- **Every LLM call gets logged** via `self._log_interaction(query, response, metadata)`.
  Keep this when adding agents; the JSONL logs are the only observability here.
- **Routing is Command-based, not edge-based.** Nodes return
  `Command(goto=..., update=state)`. The return type annotation
  (`Command[Literal["supervisor"]]`) is what LangGraph reads to infer valid
  destinations — it must match what the method actually returns.
- **State updates are copy-then-mutate**: `updated_state = dict(state)`, modify,
  return in the `Command`. `GameState` declares no reducers, so returned values
  replace rather than merge.
- **Messages are plain dicts** (`{"role", "content", "name"}`), not LangChain
  `BaseMessage` objects, despite the `GameState` type hint saying otherwise.
  `BaseAgent._get_latest_message` handles both shapes; use it rather than indexing directly.

## Rebuilding the index

No script does this. The pieces exist but are never wired together:

```python
from src.config import DOCUMENT_PATHS, CHROMA_DB_DIRECTORY
from src.data.loader import load_documents
from src.data.processing import split_documents
from src.data.vectorstore import get_vectorstore

docs = load_documents(DOCUMENT_PATHS)          # PyMuPDF, tags book + page_number
chunks = split_documents(docs)                 # 1000 chars, 200 overlap
store = get_vectorstore(chunks)                # builds only if chroma_db/ is absent
```

`get_vectorstore` returns the existing store and **ignores its `docs` argument**
whenever `chroma_db/` exists, so delete the directory first to force a rebuild.
Embeddings are `all-MiniLM-L6-v2` (384-dim) — changing the embedding model
invalidates the whole index.

## Testing

`tests/` contains two scripts, not pytest tests: no assertions, print-based output,
and they require a live Ollama plus the Chroma index. Run them directly
(`python tests/test_dice_roller.py`). Note `pyproject.toml` uses `[tool.pytest]`,
which pytest ignores — the correct table is `[tool.pytest.ini_options]`.

The one genuinely unit-testable module is `src/utils/dice.py` (pure, deterministic
apart from `random`). New tests should start there.

## Further reading

- `SPECS.md` — **the execution contract for the Claude refactor: one spec per PR**
- `docs/ARCHITECTURE.md` — how a turn flows through the system, module by module
- `docs/AGENTS.md` — per-agent contracts, prompts, and routing behavior
- `docs/RAG_PIPELINE.md` — retrieval, chunking, the index, and the unused CRAG parts
- `docs/KNOWN_ISSUES.md` — verified bugs and dead code, ranked
- `docs/REFACTOR_NOTES.md` — porting to Claude (Fable 5 / Opus 5) and modern LangGraph
