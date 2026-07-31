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
across six distinct routing prompts: mean **0.65 s**, min 0.59, max 0.70. The
throughput numbers were right.

That measurement also reported 6/6 correct routing and concluded routing was
"fast *and* accurate". The speed held; **the accuracy claim did not** — those six
cases all carried obvious keyword signals. A harder 12-case set in PR-04 put
`llama3.2:3b` at 7/12, which is why the supervisor moved to `qwen2.5:7b` (~2.7 s,
12/12). See "Model selection" below.

The real trace for a dice request, warm — **as it was before PR-04**:

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
   38 s; un-streamed, that is indistinguishable from a hang. Done in PR-06 — and
   it moved the question from *throughput* to **time-to-first-token**, which is
   prompt-eval bound, not generation bound. See #26.

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

- **PR-04** — done. The supervisor's substring-match ladder became
  `with_structured_output(Router)` over the `Router` TypedDict that had been
  sitting unused in the module. Confirmed working on both installed models.
- **PR-05** — done, though not the way this predicted. The defensive stack was
  deleted, but structured output did not replace it: a regex did. Constrained
  decoding guarantees a well-*typed* answer, not a *true* one, and the 3B kept
  inventing modifiers that were nowhere in the request (#27). Dice notation is a
  formal language; the right tool was a parser.
- **PR-08** — partly. `with_structured_output` does work locally now, so the
  grader *could* have been wired in. It was measured instead, and deleted: one
  grading call cost 31.7 s and said "yes" every time. Working is not the same as
  worth it.

Verify per model rather than assuming: JSON-schema support depends on the model
having tool-calling ability, not just on the client library. `llama3.2` and
`qwen2.5` both do.

## Model selection

The per-agent model map is worth keeping — the reasoning that produced it holds
regardless of provider. Fast model where the task is classification, stronger
model where quality shows.

| Agent | Model | Rationale |
|---|---|---|
| `supervisor` | `qwen2.5:7b` | Runs every turn — and must be *right*. See below. |
| `dice_roller` | `llama3.2:3b` | Extraction into a fixed schema. |
| `researcher` | `qwen2.5:7b` | Grounded answers over retrieved text; quality shows. |
| `dungeon_master` | `qwen2.5:7b` | Narrative coherence; the one place slowness is tolerable if streamed. |

**The supervisor row changed in PR-04, and the original reasoning was wrong.**
PR-02 gave routing to the 3B because the job is short and latency looked like the
thing to optimise. Measured across a 12-case set covering all four destinations,
`llama3.2:3b` scored **7/12** and `qwen2.5:7b` scored **12/12**; two different
prompts on the 3B scored 7 and 8, so wording was not the lever. A routing
decision is ~10 output tokens, which makes the bigger model about **1 s** more
expensive per turn — against a misroute costing ~40 s of unwanted generation, or
silence when it lands on the unimplemented `dungeon_master`. Accuracy dominates
here, not latency. `DND_MODEL_SUPERVISOR=llama3.2:3b` trades it back.

The general lesson worth carrying into PR-05 and PR-08: pick the model per *cost
of being wrong*, not per apparent size of the task.

The map landed in PR-02. `create_llm(agent_type)` resolves it, with
`DND_MODEL_<AGENT_TYPE>` / `DND_MODEL_DEFAULT` overriding it and `OLLAMA_HOST`
selecting the daemon. The old hardcoded `model_name` did not resolve at all —
Ollama tags are lowercase and carry a size suffix — so nothing in this repo could
ever have run against the default it shipped with.

## What is already done

PR-01 through PR-04 and PR-06 have landed. Of the original plan:

- **Toolchain pinned** — Python 3.12 (forced by torch's last Intel-macOS wheel),
  plus `transformers` and `numpy` constraints. See the PR-01 comment thread for
  the dependency chain.
- **Real test gate** — 88 tests, mutation-verified, none needing a daemon.
- **State contract** — `add_messages` reducer, `next_agent` removed, SQLite
  checkpointer, three real bugs fixed.
- **Model layer** — per-agent model map, env-overridable models and host, plain
  language errors for a dead daemon or a missing model. The first commit in this
  repo's history against which `create_llm()` actually returns a working client.
- **Routing** — structured output over the `Router` schema, a deterministic dice
  pre-filter, no silent `researcher` fallback, and `dice_roller` terminating
  directly. A dice request now costs 4.9 s instead of ~45 s.
- **Dungeon Master** — implemented and streaming, with world state extracted from
  each narration into `game_state`. First token ~3.6 s. The last stubbed agent.
- **Dice** — parsed deterministically, no LLM call on the common path (0.0 s,
  down from 4.6 s), and no more invented modifiers.
- **Ingestion** — `scripts/ingest.py`; the index is reproducible from the repo
  for the first time, and `load`/`build` are separate functions that say so.
- **Researcher** — real citations from chunk metadata, streamed and bounded
  answers, and a free relevance gate that triggers one query rewrite on a miss.
  A rules question went from 181 s to ~26 s.

## What remains

**Every PR in `SPECS.md` has landed.** What is left was never scheduled.

1. **The `chroma_db/` decision (#20, #23).** Gitignoring the store would stop it
   dirtying the working tree on every read *and* remove 4,778 chunks of verbatim
   rulebook text from a public repo — but it breaks out-of-the-box retrieval for
   anyone without the PDFs. Re-indexing from the CC-BY SRD 5.1 resolves both at
   once. This is a decision, not a task.
2. **Citation offsets (#28).** Cited pages are PDF page indices, several behind
   the printed page number. Worth fixing with the next re-index, since a citation
   exists to be checked.
3. **Resume a campaign.** The checkpointer works; `main.py` mints a fresh
   `thread_id` per run. One CLI flag.
4. **`src/actors/` (#14).** `Player` and `NPC` are type hints only. Now that
   `game_state` carries real world state, populating `players` has a point.

## Still-relevant LangGraph work

Not scheduled, and independent of the model layer:

- **Consolidate the agent registry.** Adding an agent means editing four places:
  node registration, `AGENT_TYPES`, the supervisor's parse chain, and
  `SUPERVISOR_PROMPT`.
- **Activate `src/actors/`.** `Player` and `NPC` are type-hint-only. The PR-03
  checkpointer makes `players`/`npcs` worth populating for the first time.
- **Use the checkpointer for real.** `main.py` generates a fresh `thread_id` per
  run. Resuming a campaign is one CLI flag away.
