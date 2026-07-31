# Refactor notes

Direction: **stay on local models via Ollama**, and modernize around them.

An earlier draft of this document planned a swap to the Anthropic API. That was
based on a misread of a tentative remark and has been withdrawn — the project
runs locally. Nothing merged so far depended on it: PR-00 through PR-03 are all
provider-agnostic.

Measurements below were taken 2026-07-31 on the target machine: Intel Core
i9-9980HK, 16 cores, 32 GB RAM, macOS. **Ollama has no GPU path on Intel Macs**
— the Radeon Pro Vega 20 is not used, so everything is CPU-only.

## The headline problem: latency, not quality

| Measurement | `llama3.2:3b` | `qwen2.5:7b` |
|---|---|---|
| Supervisor routing call (short prompt, short answer) | **5.69 s** | — |
| Generation throughput | **10.4 tok/s** | **5.1 tok/s** |

Routing was *correct* — `"roll 2d10 + 1d6"` returned `dice_roller`. The issue is
what it costs. Trace a single dice request through the current graph:

```
supervisor  (LLM call)   ~5.7s
dice_roller (LLM parse)  ~3-5s
supervisor  (LLM call)   ~5.7s     <- because dice_roller returns to supervisor
                         ───────
                         ~15s for "roll a d20"
```

`qwen2.5:7b` runs at exactly half the 3B's throughput — measured, not estimated
— so a 200-token DM narration takes about **40 seconds**. **On this hardware the
architecture is the bottleneck, not the model.** Three consequences shape the
specs:

1. **Every avoidable LLM call is worth removing.** The `dice_roller → supervisor`
   return edge doubles routing cost for the most common request type. PR-04
   already plans to fix the spurious researcher hop; it should also let
   `dice_roller` terminate directly.
2. **Not every routing decision needs a model.** A dice-notation regex
   (`\d*d\d+`) catches most dice requests deterministically, in microseconds. A
   cheap pre-filter ahead of the LLM router would remove ~5.7 s from the most
   frequent path. Fall through to the LLM only when the regex does not match.
3. **Streaming is not cosmetic here.** At 10 tok/s, a non-streamed narration
   feels like a hang. The DM agent must stream.

## Structured output works locally now

The original argument for leaving Ollama was that `with_structured_output` was
unreliable on local models. That was true when this code was written in
February 2025. It is not true with the currently pinned stack:

```
langchain-ollama 1.1.0
  with_structured_output: True   method='json_schema' (default), also
                                 'function_calling' and 'json_mode'
  bind_tools           : True
  ChatOllama.format    : Literal['', 'json'] | dict   (accepts a JSON schema)
```

This is what makes the planned improvements viable without changing provider:

- **PR-04** — the supervisor's substring-match ladder becomes
  `with_structured_output(Router)` over the `Router` TypedDict already sitting
  unused in the module.
- **PR-05** — the dice parser's defensive stack (markdown-fence stripping,
  `PQXYpqxy` placeholder detection, string-modifier coercion, description
  salvage) collapses into one strict schema. All of it exists only because a 3B
  model could not be trusted to emit clean JSON.
- **PR-08** — `src/pipelines/grader.py` depends on `with_structured_output`,
  which is why it was never wired in. It works now.

Verify per model rather than assuming: JSON-schema support depends on the model
having tool-calling ability, not just on the client library. `llama3.2` and
`qwen2.5` both do.

## Model selection

The per-agent model map is worth keeping — the reasoning that produced it holds
regardless of provider. Fast model where the task is classification, stronger
model where quality shows.

| Agent | Model | Rationale |
|---|---|---|
| `supervisor` | `llama3.2:3b` | Runs every turn; pure classification. Latency dominates. |
| `dice_roller` | `llama3.2:3b` | Extraction into a fixed schema. |
| `researcher` | `qwen2.5:7b` | Grounded answers over retrieved text; quality shows. |
| `dungeon_master` | `qwen2.5:7b` | Narrative coherence; the one place slowness is tolerable if streamed. |

`src/models/llm.py` currently hardcodes `model_name="Llama3.2"`, which **does not
resolve** — verified: `ResponseError: model 'Llama3.2' not found (404)`. Ollama
tags are lowercase and include a size (`llama3.2:3b`). This is almost certainly a
long-standing bug; nothing in the repo could ever have run against it.

## What is already done

PR-01 through PR-03 have landed. Of the original plan:

- **Toolchain pinned** — Python 3.12 (forced by torch's last Intel-macOS wheel),
  plus `transformers` and `numpy` constraints. See the PR-01 comment thread for
  the dependency chain.
- **Real test gate** — 29 tests, mutation-verified.
- **State contract** — `add_messages` reducer, `next_agent` removed, SQLite
  checkpointer, three real bugs fixed.

## What remains

In dependency order — see `SPECS.md` for the executable version.

1. **PR-02** — modernize the local model layer: fix the model name, per-agent
   map, configurable host/model via env.
2. **PR-04** — structured-output routing, plus the latency fixes above.
3. **PR-05** — strict-schema dice parsing; delete the defensive stack.
4. **PR-06** — implement `DungeonMaster.process_task`, which is still `pass`.
   This is the project's core agent and its largest gap.
5. **PR-07** — `scripts/ingest.py`, so the index is reproducible.
6. **PR-08** — researcher citations and corrective RAG.

## Still-relevant LangGraph work

Not scheduled, and independent of the model layer:

- **Consolidate the agent registry.** Adding an agent means editing four places:
  node registration, `AGENT_TYPES`, the supervisor's parse chain, and
  `SUPERVISOR_PROMPT`.
- **Activate `src/actors/`.** `Player` and `NPC` are type-hint-only. The PR-03
  checkpointer makes `players`/`npcs` worth populating for the first time.
- **Use the checkpointer for real.** `main.py` generates a fresh `thread_id` per
  run. Resuming a campaign is one CLI flag away.
