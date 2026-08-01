# AI Dungeon Master

A D&D 5e Dungeon Master built as a LangGraph multi-agent system. A supervisor
routes each turn to a narrator, a rules researcher, or a dice roller. Rules
answers are retrieved from the SRD 5.1 and cite the passage they came from.

**Everything runs locally through [Ollama](https://ollama.com).** No API keys, no
data leaving the machine.

```
Ask a D&D question: I push open the heavy iron door and step into the crypt.

[dungeon_master] You push open the heavy iron door, feeling the cold draft from
within. Candles flicker along the walls, casting eerie shadows across ancient
tombstones. A low murmur of whispers seems to emanate from the far end.

You see a narrow path leading deeper into the darkness.

Ask a D&D question: how does sneak attack work for rogues

[researcher] Sneak Attack lets a rogue deal an extra 1d6 damage to one creature
they hit once per turn, if they have advantage on the attack roll or another
enemy of the target is within 5 feet. The attack must use a finesse or ranged
weapon.

---
**Passages consulted:**
- SRD 5.1, Class Features: Sneak Attack

Ask a D&D question: roll 2d6+3 for damage

[dice_roller] 🎲 Rolled 2d6 + 3 for damage: **12** (rolled [4, 5])
```

## Getting started

Needs **Python 3.12 specifically** — see [`CLAUDE.md`](CLAUDE.md) for the two
independent constraints that pin it.

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

ollama serve &                # or launch Ollama.app
ollama pull llama3.2:3b       # note the tag: "Llama3.2" does not resolve
ollama pull qwen2.5:7b

python scripts/ingest.py      # build the vector index, ~35s, no network
python main.py
```

Type `quit` or `exit` to leave.

## How a turn works

```
                       supervisor
                     /     |      \
          dungeon_master  researcher  dice_roller
                     \     |      /
                          END
```

Every worker ends the turn — nothing routes back — so a turn costs exactly one
routing decision and one worker.

| Agent | Job | Model |
|---|---|---|
| `supervisor` | Route the turn | `qwen2.5:7b` |
| `dungeon_master` | Narrate the world's response, stream it, track location and inventory | `qwen2.5:7b` |
| `researcher` | Answer rules questions from the SRD, with citations | `qwen2.5:7b` |
| `dice_roller` | Parse and roll dice | `llama3.2:3b` |

Dice requests never reach a model: the notation is read with a parser, so
`roll 2d6+3` is instant and exact. The supervisor is asked only when a request
is genuinely ambiguous.

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
| Routing | ~2.7 s |
| Narration, first token | ~3.6 s, then streams |
| Rules answer | ~26 s |

Generation throughput is the bottleneck, not the architecture: 11.4 tok/s on the
3B, 5.3 tok/s on the 7B. Narration and rules answers stream for that reason.

## Layout

| Path | |
|---|---|
| `main.py` | REPL; streams the graph token by token |
| `src/agents/` | `supervisor`, `dungeon_master`, `researcher`, `dice_roller`, `base_agent` |
| `src/graph/` | `StateGraph` wiring and the `GameState` contract |
| `src/models/llm.py` | The single LLM factory — per-agent models, env overrides |
| `src/data/` | SRD and PDF loaders, chunking, the Chroma store |
| `src/prompts/` | Every system prompt, as module constants |
| `src/utils/dice.py` | Pure dice parser and roller |
| `scripts/ingest.py` | Builds the vector index |
| `corpus/srd/` | The vendored SRD 5.1 corpus |
| `docs/` | Architecture, agents, RAG pipeline, known issues |

## Tests

```bash
pytest                        # 218 tests
pytest -m "not integration"   # unit only, no dependency stack
pytest -m slow                # includes a real embedding round-trip
```

Nothing in the suite calls a model, so the gate stays fast and runs offline.

## Docs

- [`CLAUDE.md`](CLAUDE.md) — orientation and conventions
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — how a turn flows, module by module
- [`docs/AGENTS.md`](docs/AGENTS.md) — per-agent contracts and prompts
- [`docs/RAG_PIPELINE.md`](docs/RAG_PIPELINE.md) — retrieval, chunking, corrective RAG
- [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md) — verified bugs and dead code, with a status ledger
- [`docs/REFACTOR_NOTES.md`](docs/REFACTOR_NOTES.md) — direction and measured performance
- [`docs/SPECS.md`](docs/SPECS.md) — the refactor's execution contract, one spec per PR

## Licence

Project code: see the repository licence. Game content in `corpus/srd/` is from
the SRD 5.1 by Wizards of the Coast, licensed CC-BY-4.0 — attribution in
[`corpus/README.md`](corpus/README.md).
