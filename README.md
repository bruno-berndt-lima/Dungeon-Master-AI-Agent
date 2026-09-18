# AI Dungeon Master

A D&D 5e Dungeon Master built on LangGraph. The model narrates; a
deterministic engine resolves every check, attack, and hit point; the SRD 5.1
answers rules questions with citations. Everything runs locally through
[Ollama](https://ollama.com) — no API keys, no data leaving the machine.

```
> /join rogue as Kara
[intake] Kara joins the party — Kara (Halfling Rogue 1): AC 14, 9/9 HP.

> I creep toward the guard, keeping to the shadows.
[dungeon_master] The torchlight ends a pace short of the wall, and you fold
yourself into the dark beyond it. Kara, give me a Dexterity (Stealth) check.

> /roll 15
[intake] Kara: Dexterity (Stealth) check: d20 15 + 7 = 22 vs DC 13 — success
[dungeon_master] The guard yawns and turns back to the brazier. You are past him.

> /rules how does sneak attack work
[researcher] Sneak Attack lets a rogue deal an extra 1d6 damage once per turn…
---
**Passages consulted:**
- SRD 5.1, Class Features: Sneak Attack
```

## Getting started

Needs **Python 3.11 or newer** (3.12 and 3.13 verified). No torch: embeddings
are served by the same Ollama daemon as the chat models.

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

ollama serve &                # or launch Ollama.app
ollama pull llama3.2:3b       # note the tag: "Llama3.2" does not resolve
ollama pull qwen2.5:7b
ollama pull all-minilm        # embeddings run through Ollama too

python scripts/ingest.py      # build the vector index, ~2 min on CPU, no network
python main.py
```

Type `quit` or `exit` to leave. The campaign is saved as you play:

```bash
python main.py --list               # every campaign in game_state.db
python main.py --thread 3f9a1c2e    # pick one up where it left off
```

## How a turn works

```
intake ──▶ dungeon_master ⇄ tools ──▶ END        play
   ├─────▶ researcher ──────────────▶ END        /rules
   └─────────────────────────────────▶ END        /roll, /join, /party, bare dice
```

`intake` routes in code — no model decides where a turn goes. The Dungeon
Master is called with its tools bound (`request_check`, `attack`,
`start_encounter`, `lookup_rules`, …), runs at most four tool round trips, and
always ends with narration. **The model never produces a number:** rolls,
DCs, damage, and hit points come from `src/engine/`, and the DM narrates what
the tools report. A requested roll ends the DM's turn; the player's `/roll`
resolves it against their sheet and the DM narrates the outcome.

| Node | Job | Model |
|---|---|---|
| `intake` | Route the turn; roll bare dice; resolve pending rolls | none |
| `dungeon_master` | Narrate, choose tools | `qwen2.5:7b` |
| `tools` | Run the tools against the engine | none |
| `memory` | Fold old messages into the campaign journal | `qwen2.5:7b`, only when the transcript is long |
| `researcher` | Answer `/rules` questions from the SRD, with citations | `qwen2.5:7b` |

## The corpus

Retrieval runs over the **System Reference Document 5.1**, published by Wizards
of the Coast under **CC-BY-4.0**. It ships in [`corpus/srd/`](corpus/README.md) —
2,313 entries indexing to 3,082 chunks — so the index rebuilds from the
repository alone, with no downloads.

One document per *entry* rather than per page, which is why answers cite
`SRD 5.1, Monsters: Goblin` instead of a page number you cannot check.

If you own the rulebooks, `scripts/ingest.py --source rulebooks` builds a wider
index from your own PDFs into a gitignored directory. See
[`Documents/README.md`](Documents/README.md).

## Performance

Measured on an Intel i9-9980HK — **Ollama has no GPU path on Intel Macs**, so
this is CPU-only. Expect an Apple Silicon or CUDA machine to be far quicker.

| | |
|---|---|
| Dice roll | instant (no model call) |
| Routing | 0 s — code, since PR-18 |
| Narration, first token | 2.5–6 s warm (the static prompt prefix is cached), then shown |
| Rules answer | ~26 s |

Generation throughput is the bottleneck, not the architecture: 11.4 tok/s on the
3B, 5.3 tok/s on the 7B. Narration and rules answers stream for that reason.

## Layout

| Path | |
|---|---|
| `main.py` | REPL; `--thread` / `--list`; streams the graph token by token |
| `src/agents/` | `dungeon_master` (the tool loop), `researcher`, `base_agent` |
| `src/engine/`, `src/srd/`, `src/tools/` | The deterministic 5e engine, the SRD as data, and the DM's tools over both |
| `src/graph/` | `intake` routing, `StateGraph` wiring, the `GameState` contract, campaigns |
| `src/models/llm.py` | The single LLM factory — per-agent models, env overrides |
| `src/data/` | SRD and PDF loaders, chunking, the Chroma store |
| `src/prompts/` | Every system prompt, as module constants |
| `src/utils/dice.py` | Pure dice parser and roller |
| `scripts/ingest.py` | Builds the vector index |
| `corpus/srd/` | The vendored SRD 5.1 corpus |
| `docs/` | Architecture, agents, RAG pipeline, known issues |

## Tests

```bash
pytest                        # 614 tests
pytest -m "not integration"   # unit only, no dependency stack
pytest -m slow                # includes a real embedding round-trip (needs the daemon)
```

Nothing in the suite calls a model, so the gate stays fast and runs offline.

## Docs

- [`CLAUDE.md`](CLAUDE.md) — orientation and conventions
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — how a turn flows, module by module
- [`docs/DM.md`](docs/DM.md) — the Dungeon Master's tool loop
- [`docs/ENGINE.md`](docs/ENGINE.md) and [`docs/TOOLS.md`](docs/TOOLS.md) — the engine and the tools over it
- [`docs/RAG_PIPELINE.md`](docs/RAG_PIPELINE.md) — retrieval, chunking, corrective RAG
- [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md) — verified bugs and dead code, with a status ledger
- [`docs/REFACTOR_NOTES.md`](docs/REFACTOR_NOTES.md) — direction and measured performance
- [`docs/SPECS.md`](docs/SPECS.md) — the refactor's execution contract, one spec per PR

## Licence

Project code: see the repository licence. Game content in `corpus/srd/` is from
the SRD 5.1 by Wizards of the Coast, licensed CC-BY-4.0 — attribution in
[`corpus/README.md`](corpus/README.md).
