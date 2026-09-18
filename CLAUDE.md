# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

An AI Dungeon Master for D&D 5e: a LangGraph multi-agent system where a supervisor
routes player input to specialist agents (narrator, rules researcher, dice roller).
Rules retrieval is RAG over the **SRD 5.1** (CC-BY-4.0), which ships in
`corpus/srd/` and is indexed in a local ChromaDB.
All inference currently runs locally through Ollama.

The project **runs**: all four agents are implemented, routing is schema-constrained,
and narration streams. The deterministic 5e engine under `src/engine/` is complete and
tested but not yet wired into the agents (that is Phase 2 of `docs/ROADMAP.md`).

## Running it

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
ollama serve &              # or launch Ollama.app
ollama pull llama3.2:3b     # note the tag: "Llama3.2" does NOT resolve
ollama pull qwen2.5:7b
ollama pull all-minilm      # embeddings — also served by the daemon
python main.py              # interactive REPL; type "quit" or "exit" to leave
python main.py --list       # campaigns in game_state.db
python main.py --thread ID  # resume one (or name a new one)
```

**Python 3.11 or newer.** `src/agents/supervisor.py` uses `Literal[*ROUTING_OPTIONS]`
(PEP 646), a syntax error before 3.11. Verified on 3.12 and 3.13 (2026-09-18).

It used to be "3.12 exactly": embeddings ran through `sentence-transformers`,
which needs torch, and on Intel macOS torch's last x86_64 wheel only had a
cp312 tag. PR-12 moved embeddings to the Ollama daemon, so there is no torch,
no `transformers`, and no numpy pin in `requirements.txt` any more.

Requires a running Ollama daemon. **Build the index once before first use** —
`python scripts/ingest.py` takes ~2 min on CPU and needs nothing but the repository and
the daemon:

```bash
python scripts/ingest.py        # corpus/srd/ -> chroma_db/
```

`chroma_db/` is a build artifact and gitignored, not committed. The commercial
rulebook PDFs are optional; they only buy wider coverage — see
`Documents/README.md` and "Rebuilding the index" below.

**Expect generation to be slow.** Ollama has no GPU path on Intel Macs, so this
is CPU-only: **11.4 tok/s** on `llama3.2:3b`, **5.3 tok/s** on `qwen2.5:7b` — a
200-token answer takes 18 s and 38 s respectively. Routing is not the problem
(~0.65 s warm); output tokens are. The first call to each model also pays a cold
load of 5–11 s. See `docs/KNOWN_ISSUES.md` #24.

## Layout

| Path | Role |
|---|---|
| `main.py` | REPL; `--thread` / `--list` pick a campaign, streams the compiled graph token by token |
| `src/graph/campaigns.py` | Campaigns as checkpointer threads: ids, listing, recap, first-turn seeding |
| `src/config.py` | Chroma dir, PDF paths, embedding model name |
| `src/graph/game_orchestrator.py` | Builds the `StateGraph`, registers agent nodes |
| `src/graph/game_state.py` | `GameState` TypedDict, default factory, and the `get_party` / `put_party` style accessors that keep engine models as JSON in state |
| `src/agents/` | `base_agent` (ABC), `supervisor`, `dungeon_master`, `researcher`, `dice_roller` |
| `src/engine/` | The 5e engine, no LLM: `character` (sheets), `checks` (d20 resolution), `combatant`, `combat` (encounters), `pregens`. See `docs/ENGINE.md` |
| `src/srd/` | The SRD JSON as data: `monster()`, `spell()`, `equipment()`, `condition()` with fuzzy names; `bestiary.summon()` |
| `src/tools/` | The DM's tools over the engine — one function per tool, pydantic args, `run_tool` never raises. See `docs/TOOLS.md` |
| `src/data/` | `srd_loader` (JSON, default), `loader` (PDF), `processing`, `vectorstore` |
| `corpus/srd/` | The vendored SRD 5.1 corpus. **Committed** — see `corpus/README.md` |
| `scripts/ingest.py` | Rebuilds `chroma_db/` from the PDFs; `--rebuild`, `--dry-run` |
| `src/pipelines/` | `grader`, `rewriter`, `generator` — corrective-RAG parts, currently unused |
| `src/models/llm.py` | `create_llm(agent_type)` and `create_embedding_model()` — the single provider boundary; per-agent model map, env overrides |
| `src/prompts/prompts.py` | All system prompts, as module-level string constants |
| `src/utils/dice.py` | Pure dice notation parser + roller (no LLM) |
| `src/utils/llm_logger.py` | Appends every agent call to `logs/llm_interactions/*.jsonl` |
| `Documents/` | Where the three 5e PDFs go. **Gitignored** — supply your own; see `Documents/README.md` |
| `chroma_db/` | Persisted vector store, 3082 chunks, 384-dim (`all-minilm`). **Gitignored — build it** |

## Conventions to follow

- **One LLM factory.** Every agent calls `create_llm(self.agent_type)` from
  `src/models/llm.py` — after `super().__init__(...)`, so `agent_type` is set. The
  model is chosen there, per role, from `AGENT_MODELS`; a new agent type not in
  that map falls back to `DEFAULT_MODEL`. Change models, host, or provider in that
  one file, never in an agent. Overrides without editing code:
  `DND_MODEL_<AGENT_TYPE>`, `DND_MODEL_DEFAULT`, `DND_EMBEDDING_MODEL`, `OLLAMA_HOST`.
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
python scripts/ingest.py                  # SRD 5.1 -> chroma_db/  (first run)
python scripts/ingest.py --rebuild        # replace an existing index
python scripts/ingest.py --dry-run        # chunk without embedding
```

The corpus is committed under `corpus/srd/`, so this needs no PDFs and no
network beyond the local daemon. ~2 min on CPU (109 s measured), 3,082 chunks.

**Two corpora, two indexes.** The SRD is a subset of the published books, so if
you own them:

```bash
python scripts/ingest.py --source rulebooks --rebuild   # -> chroma_db_full/
DND_CHROMA_DIR=chroma_db_full python main.py
```

Both indexes are gitignored build artifacts; `chroma_db/` is the default.
`corpus/README.md` has the coverage comparison.

Under the hood, SRD path: `load_srd_documents` renders **one document per entry**
(monster, spell, rule section) and chunks each one itself, re-heading every piece
with the entry name — Chroma embeds `page_content` and never metadata, so a piece
without its title cannot be found by name. PDF path: `load_documents` (PyMuPDF,
tags `book` + `page_number`) → `split_documents`.

`load_vectorstore()` reads and `build_vectorstore(docs, rebuild=...)` writes.
Building over an existing store is refused, because Chroma appends and would
duplicate every chunk. Embeddings are `all-minilm` (all-MiniLM-L6-v2, 384-dim)
served by Ollama — `EMBEDDING_MODEL` in `src/models/llm.py`, overridable with
`DND_EMBEDDING_MODEL`; changing it invalidates the whole index.

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

- `docs/ROADMAP.md` — **where the project is going next: engine, tool-using DM, Discord** (PR-11 onward)
- `docs/SPECS.md` — the refactor's execution contract, PR-00 to PR-10, all landed
- `docs/ARCHITECTURE.md` — how a turn flows through the system, module by module
- `docs/ENGINE.md` — the deterministic 5e engine: models, rules covered, what is left out
- `docs/TOOLS.md` — the DM's tools: what each does and writes, and why `run_tool` never raises
- `docs/AGENTS.md` — per-agent contracts, prompts, and routing behavior
- `docs/RAG_PIPELINE.md` — retrieval, chunking, the index, and the unused CRAG parts
- `docs/KNOWN_ISSUES.md` — verified bugs and dead code, ranked
- `docs/REFACTOR_NOTES.md` — direction, measured local-model performance, and what's left
