# Refactor notes

Written for the planned move off Ollama/Llama 3.2 onto Claude, plus the LangGraph
modernization that goes with it. Model facts below are current as of 2026-07-31.

## Model choice

| Model | ID | Context | $/1M in | $/1M out |
|---|---|---|---|---|
| Claude Fable 5 | `claude-fable-5` | 1M | $10.00 | $50.00 |
| Claude Opus 5 | `claude-opus-5` | 1M | $5.00 | $25.00 |
| Claude Sonnet 5 | `claude-sonnet-5` | 1M | $3.00 (intro $2.00 through 2026-08-31) | $15.00 (intro $10.00) |
| Claude Haiku 4.5 | `claude-haiku-4-5` | 200K | $1.00 | $5.00 |

Fable 5 is Anthropic's most capable widely released model — built for demanding
reasoning and long-horizon agentic work. It is genuinely a good match for the
*narrative* agent: sustained world consistency across a long campaign is exactly the
long-horizon coherence it's strongest at.

Two honest caveats before making it the default everywhere:

- **Price.** It sits above the Opus tier. This graph makes 2–3 LLM calls per player
  turn today (supervisor routing, plus the agent, plus a second supervisor hop after
  dice). Routing and dice parsing are trivial classification/extraction tasks; paying
  $50/1M output for them is waste.
- **Latency.** Fable 5's thinking is always on and single requests on hard tasks can
  run for minutes at higher effort. That is fine for an overnight agent and wrong for
  a synchronous `input()` REPL where a player is waiting. If you keep Fable 5 on the
  interactive path, run it at `low` or `medium` effort and stream.

**Recommended split** — one factory function, one model per role:

```python
# src/models/llm.py
MODELS = {
    "dungeon_master": "claude-fable-5",   # narrative quality, long-horizon coherence
    "researcher":     "claude-opus-5",    # grounded rules answers over retrieved text
    "supervisor":     "claude-haiku-4-5", # routing is classification
    "dice_roller":    "claude-haiku-4-5", # parsing is extraction
}
```

If you'd rather keep it to one model, `claude-opus-5` is the workhorse choice at half
Fable's price with the same 1M context and feature set.

## API differences that hit this codebase directly

These are breaking on Fable 5 / Opus 5 / Sonnet 5 and will bite during the port:

**1. `temperature` is rejected.** `src/models/llm.py` currently does
`create_llm(model_name="Llama3.2", temperature=0)`. On Fable 5, Opus 5, and Opus 4.7/4.8,
sending `temperature`, `top_p`, or `top_k` returns a 400; Sonnet 5 rejects non-default
values. Drop the parameter from the factory signature entirely. Steer determinism with
`effort: "low"` and a tighter prompt instead — `temperature=0` never guaranteed
identical output anyway.

**2. Thinking configuration.** On Fable 5, thinking is always on — omit the `thinking`
parameter entirely; an explicit `{"type": "disabled"}` returns a 400. On Opus 5,
thinking is on by default and can be disabled only at effort `high` or below. Control
depth with `output_config: {"effort": "low"|"medium"|"high"|"xhigh"|"max"}` rather than
a token budget; `budget_tokens` is removed and 400s.

**3. `stop_reason: "refusal"`.** Fable 5 and Opus 5 run safety classifiers and can
decline a request with an HTTP 200, an empty or partial `content`, and
`stop_reason == "refusal"`. Classifier targets are bio and cyber content, so D&D
combat narration is very unlikely to trip them — but code that reads `content[0]`
unconditionally breaks if it ever does. Check `stop_reason` before reading content.
For a fallback, the server-side `fallbacks: "default"` parameter (beta header
`server-side-fallback-2026-07-01`) retries on a substitute model in the same call.

**4. Fable 5 requires 30-day data retention.** It is not available to organizations
configured for zero data retention — such requests 400 on every call regardless of
payload. Check the org setting before debugging a request body.

**5. No assistant prefill.** A last-assistant-turn prefill 400s. This codebase doesn't
use prefills, so nothing to change — worth knowing if you add one.

## LangChain vs. the Anthropic SDK

`langchain-anthropic`'s `ChatAnthropic` is the smallest diff: it drops into the
existing LCEL chain in `ResearcherAgent` unchanged. Two things to verify against the
version you install rather than assume:

- whether it forwards `temperature` by default (if so, that's an immediate 400 — see #1)
- whether it exposes `output_config.effort` and adaptive thinking, or whether you need
  to pass them through `model_kwargs`

If effort control turns out to be awkward through the LangChain wrapper, the pragmatic
split is: keep LCEL for the researcher's retrieval chain, and call the `anthropic` SDK
directly from the supervisor and dice roller, where you want tight control over
structured output and effort anyway. `src/models/llm.py` is already the single
provider boundary, so this stays contained.

Keep `sentence-transformers` / `all-MiniLM-L6-v2` for embeddings regardless. The
committed `chroma_db/` is built from its 384-dim vectors; changing the embedder means
re-indexing three books. The chat-model swap touches nothing in the index.

## What the swap fixes for free

Three of the uglier parts of this codebase exist only because the local model couldn't
be trusted to emit structure.

**Supervisor routing** (`docs/KNOWN_ISSUES.md` #5). Replace the substring-match chain
with structured output. The `Router` TypedDict already sitting unused in
`supervisor.py` is exactly the right schema:

```python
class Router(TypedDict):
    next: Literal["dungeon_master", "researcher", "dice_roller", "FINISH"]
```

Bind it with `with_structured_output(Router)` (or a strict tool with
`"strict": True` and `additionalProperties: false`), and the whole
`if "dice_roller" in text` ladder plus the catch-all `researcher` fallback goes away.
The routing prompt should also be told the request may already be satisfied, which
fixes the spurious post-dice research turn (#6).

**Dice parsing** (`src/agents/dice_roller.py:_parse_dice_request`). The markdown-fence
stripping, the `PQXYpqxy` placeholder detection, the string-modifier coercion, the
`for`/`to check` description salvage — all of it is scar tissue from coaxing JSON out
of a small local model. A strict tool schema replaces the entire method body with one
call. Keep `DiceRoller` doing the actual rolling; the LLM-parses/code-rolls split is
the right design and should survive.

**Retrieval grading** (`src/pipelines/grader.py`). It depends on
`with_structured_output`, which is why it was never wired in against Ollama. It works
reliably against Claude, which makes the corrective-RAG loop in `src/pipelines/`
finally viable — grade retrieved docs, and on a miss run `rewriter.py` and retry.

## LangGraph modernization

Independent of the provider, worth doing in the same pass:

**Add reducers to `GameState`.** `messages: Annotated[Sequence[BaseMessage], add_messages]`
removes the copy-the-whole-list-and-return-it pattern from every agent and fixes the
silent-history-drop hazard (#7).

**Pick one message representation.** Today the type hint says `BaseMessage` and the
runtime holds plain dicts. Go with `BaseMessage` throughout and delete the dual-shape
handling in `BaseAgent._get_latest_message`.

**Add a checkpointer.** A `SqliteSaver` gives persistent campaign state across
sessions, which is the difference between a demo and something you'd actually play.
It also makes the six never-written state keys (`players`, `npcs`, `turn_order`,
`current_speaker`) worth populating, and gives the `src/actors/` classes their first
real use.

**Consolidate the agent registry.** Adding an agent currently means editing four
places: the node registration, `AGENT_TYPES`, the supervisor's parse chain, and
`SUPERVISOR_PROMPT`. One registry that generates all four is a small change with an
outsized payoff.

**Drop the `next_agent` mirror.** `Command(goto=...)` is the real routing channel;
`state["next_agent"]` duplicates it and the two can diverge.

**Stream the DM.** Narrative output is the one place a player will feel latency.
Stream it, especially if the DM runs on Fable 5.

## Order

1. Pin Python ≥3.11 and pin dependency versions — the current unpinned set won't
   reproduce (`docs/KNOWN_ISSUES.md` #2, #19).
2. Swap `src/models/llm.py` to Claude, per-agent model map, no `temperature`.
3. Convert supervisor routing to structured output; delete the substring ladder.
4. Convert dice parsing to a strict tool; delete `_parse_dice_request`'s defensive stack.
5. **Implement `DungeonMaster.process_task`** — the actual point of the project. Everything
   it needs (prompt, factory, state, logging, `Command` pattern) is already in place.
6. Add `add_messages` reducers and a checkpointer.
7. Write `scripts/ingest.py` so the index is reproducible.
8. Wire in `grader.py` + `rewriter.py` for corrective RAG.
9. Surface `book` / `page_number` in the researcher's context so citations actually work.

Steps 1–5 get you a system that runs end to end. Everything after that is quality.
