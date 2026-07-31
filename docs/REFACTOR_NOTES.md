# Refactor notes

Direction: **stay on local models via Ollama**, and modernize around them.

An earlier draft of this document planned a swap to the Anthropic API. That was
based on a misread of a tentative remark and has been withdrawn — the project
runs locally. Nothing merged so far depended on it: PR-00 through PR-03 are all
provider-agnostic.

Measurements below were taken 2026-07-31 on the target machine: Intel Core
i9-9980HK, 16 cores, 32 GB RAM, macOS. **Ollama has no GPU path on Intel Macs**
— the Radeon Pro Vega 20 is not used, so everything is CPU-only.

## The headline problem: tokens generated, not calls made

| Measurement | `llama3.2:3b` | `qwen2.5:7b` |
|---|---|---|
| Cold load (first call after start or eviction) | ~5 s | ~11 s |
| Warm routing call (system prompt + one turn) | **0.65 s** | — |
| Generation throughput | **11.4 tok/s** | **5.3 tok/s** |

**Correction to an earlier draft of this file.** It reported a 5.69 s supervisor
routing call and built a whole optimisation program on it. That measurement was a
**cold model load** taken on a first call, not steady state. Re-measured warm
across six distinct routing prompts: mean **0.65 s**, min 0.59, max 0.70 — and
6/6 routed correctly (`"roll 2d10 for damage"` → `dice_roller`, `"I open the
door"` → `dungeon_master`, `"how does grappling work"` → `researcher`). Routing
on this hardware is fast and accurate. The throughput numbers were right.

The real trace for a dice request, warm:

```
supervisor  (route)      ~0.7s
dice_roller (LLM parse)  ~1-2s
supervisor  (route)      ~0.7s   <- because dice_roller returns to supervisor
researcher  (GENERATE)   ~40s    <- routed on the roll result; answers nothing asked
                         ───────
                         ~43s, of which 93% is one unwanted generation
```

**Output tokens are the whole cost.** A routing call emits three tokens; a
researcher answer emits two hundred at 5.3 tok/s. Three consequences shape the
specs:

1. **Never generate something nobody asked for.** The expensive part of #6 is not
   the extra routing hop — it is the ~40 s answer the supervisor triggers by
   routing on the roll *result*. PR-04 letting `dice_roller` terminate directly
   removes an entire generation, not 0.65 s.
2. **A regex pre-filter is about determinism, not speed.** `\d*d\d+` classifies
   dice requests exactly, where a 3B model classifies them *usually*. Worth doing
   for #5 — but it saves 0.65 s, so do not sell it as a latency fix.
3. **Streaming is not cosmetic here.** At 5.3 tok/s a 200-token narration takes
   38 s; un-streamed, that is indistinguishable from a hang. The DM agent must
   stream. This is the single largest perceived-latency win available.

Both models stay resident simultaneously (2.6 GB + 5.1 GB against 32 GB, verified
via `/api/ps`), so the per-agent model map costs no swap penalty here. On a
smaller machine it would, and one model for all four roles would be better.

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

The map landed in PR-02. `create_llm(agent_type)` resolves it, with
`DND_MODEL_<AGENT_TYPE>` / `DND_MODEL_DEFAULT` overriding it and `OLLAMA_HOST`
selecting the daemon. The old hardcoded `model_name` did not resolve at all —
Ollama tags are lowercase and carry a size suffix — so nothing in this repo could
ever have run against the default it shipped with.

## What is already done

PR-01 through PR-03 have landed. Of the original plan:

- **Toolchain pinned** — Python 3.12 (forced by torch's last Intel-macOS wheel),
  plus `transformers` and `numpy` constraints. See the PR-01 comment thread for
  the dependency chain.
- **Real test gate** — 29 tests, mutation-verified.
- **State contract** — `add_messages` reducer, `next_agent` removed, SQLite
  checkpointer, three real bugs fixed.
- **Model layer** — per-agent model map, env-overridable models and host, plain
  language errors for a dead daemon or a missing model. The first commit in this
  repo's history against which `create_llm()` actually returns a working client.

## What remains

In dependency order — see `SPECS.md` for the executable version.

1. **PR-04** — structured-output routing, and let `dice_roller` terminate so the
   graph stops paying for an unwanted researcher generation.
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
