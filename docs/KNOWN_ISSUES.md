# Known issues

Findings from a full read of every source file. Each is traceable to a specific line;
none are speculative. Ordered by impact.

## Status ledger

Numbering is stable — `docs/SPECS.md` references these IDs, so nothing is renumbered
as items close.

| # | Issue | Status |
|---|---|---|
| 1 | `DungeonMaster.process_task` is a stub | **fixed** (PR-06) |
| 2 | Needs Python 3.11+ | **fixed** (PR-01) |
| 3 | Return annotations disagree with returns | **fixed** (PR-03) |
| 4 | `game_state` dict/string collision | **fixed** (PR-03) |
| 5 | Supervisor routes by substring match | **fixed** (PR-04) |
| 6 | No terminal condition after a dice roll | **fixed** (PR-04) |
| 7 | `GameState` declares no reducers | **fixed** (PR-03) |
| 8 | `dice_roller` mutates shared caller state | **fixed** (PR-03) |
| 9 | `parse_dice_string` only splits on `+` | **fixed** (PR-05) |
| 10 | `Chroma` imported twice | **fixed** (PR-07) |
| 11 | `get_vectorstore` ignores its argument | **fixed** (PR-07) |
| 12 | No ingestion entry point | **fixed** (PR-07) |
| 13 | `src/pipelines/` unused | **fixed** (PR-08) — rewriter wired, generator and grader deleted |
| 14 | `src/actors/` unused | open — deferred |
| 15 | Unused declarations | **fixed** — `create_json_llm` (PR-02), `Router`/`State` (PR-04), `DiceRollRequest` (PR-05), `format_docs` (PR-08) |
| 16 | State keys never written | partial — `last_response` (PR-03) and `game_state` (PR-06) written; three remain |
| 17 | `tests/` are not tests | **fixed** (PR-01) |
| 18 | pytest config in the wrong table | **fixed** (PR-01) |
| 19 | `requirements.txt` unpinned | **fixed** (PR-01) |
| 20 | Large binaries committed | **fixed** (PR-10) — history rewritten, indexes gitignored; repo 256 MB → <1 MB |
| 21 | `env_activation.txt` is Windows-only | **fixed** — file deleted |
| 22 | `create_llm` model name does not resolve | **fixed** (PR-02) |
| 23 | Chroma dirties the repo on read | **fixed** (PR-10) — `chroma_db/` is gitignored, so its read-churn is invisible to git |
| 24 | Generation throughput dominates | **mitigated** — #6 removed (PR-04), narration streams (PR-06) |
| 25 | A 3B model is not accurate enough to route | **fixed** (PR-04) |
| 26 | Time-to-first-token is dominated by prompt evaluation | **mitigated** — DM (PR-06) and researcher (PR-08) both tuned |
| 27 | A local model invents dice modifiers | **fixed** (PR-05) |
| 28 | Cited page numbers are PDF pages, not printed pages | **moot on the default corpus** (PR-09) — SRD chunks cite by entry name; still applies to `chroma_db_full/` |

Seven items were found after the initial audit and are described at the bottom
of this file: #22 through #28.

## Blocking

### 1. `DungeonMaster.process_task` is a stub

`src/agents/dungeon_master.py:23-25` — the body is `pass`, so the node returns `None`
instead of a `Command`. The node is registered in the graph and the supervisor can
route to it, so any narrative input reaches a dead end. This is the project's core
agent.

**Fixed in PR-06.** It narrates, streams, terminates, and lifts durable facts
(location, inventory, effects) out of the narration into `game_state`, which is
fed back as a one-line briefing on the next turn. A test asserts the node returns
a `Command` even when the model call fails — returning `None` is the failure mode
that defined this issue.

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

**Fixed in PR-04.** Routing is now `with_structured_output(Router)`, so `next` is
a `Literal` the model cannot violate — there is no text to substring-match. A
deterministic pre-filter handles unambiguous dice notation ahead of the model.
The catch-all `goto = "researcher"` is gone: an unroutable turn ends with an
explicit message instead of a confident RAG answer to a question nobody asked.

### 6. No terminal condition after a dice roll

`dice_roller` returns to `supervisor`, which then re-reads the full history. The newest
message is now the roll result, so the supervisor routes on the *result* rather than
the original request. `logs/llm_interactions/llm_log_2025-03-31.jsonl` captures this
exactly: `roll 2d10 + 1d6` → `dice_roller` → `supervisor` → `researcher`, which then
answers a rules question nobody asked. `SUPERVISOR_PROMPT` has no concept of "already
satisfied."

**Fixed in PR-04**, two ways. `dice_roller` now terminates instead of returning
to the supervisor, and the supervisor routes on `current_task` — what the player
asked this turn — rather than the message tail, so an agent's own output can
never become the thing being routed. Verified: `roll 2d10 + 1d6` produces exactly
one roll and ends, in 4.9 s rather than ~45 s.

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

**Fixed in PR-05.** The parser splits on signed terms, skips flat modifiers, and
**raises** on anything it cannot represent — a subtracted dice term (`2d6-1d4`
would otherwise roll as `2d6+1d4`), a zero quantity or die size, or junk. Two
tests that pinned the old lenient behaviour (`""` returning `[]`) were changed
deliberately: silently returning "no dice" for input we did not understand is the
same failure as defaulting to `1d20`. It looks like success.

### 10. `Chroma` imported twice

`src/data/vectorstore.py:3-4` imports `Chroma` from `langchain_community.vectorstores`
then immediately from `langchain_chroma`. The second wins. Remove the first.

### 11. `get_vectorstore` silently ignores its argument

`src/data/vectorstore.py:12-16` — when `chroma_db/` exists, `docs` is discarded and the
existing store is returned. There's no incremental-add path and no way to tell from the
call site whether documents were indexed. `ResearcherAgent` exploits this by passing
`[]`. Split into `load_vectorstore()` and `build_vectorstore(docs)`.

**Fixed in PR-07**, exactly that split. Two silent failures now raise: loading a
missing index (which used to return an empty but usable store, so a missing index
looked like a working one that retrieved nothing) and building over an existing
one (Chroma appends — verified, 6 chunks became 7 — so it would have duplicated
every chunk). `ResearcherAgent` calls `load_vectorstore()`.

## Dead code

### 12. No ingestion entry point

`load_documents` and `split_documents` are complete and correct but imported nowhere.
The committed `chroma_db/` was built out-of-band and cannot be reproduced from the repo
without writing the script yourself.

**Fixed in PR-07** — `scripts/ingest.py`, with `--rebuild`, `--dry-run`, and a
clear message naming any missing PDF. Writing it also surfaced a second dead
import: `src/data/processing.py` used `langchain.text_splitter`, removed in
LangChain 1.x. Because nothing imported the module, that break never failed a
test run.

### 13. The entire `src/pipelines/` package is unused

`create_retrieval_grader`, `create_question_rewriter`, `create_rag_chain` — no importers.
This is a complete corrective-RAG loop that was never wired in. `generator.py`
additionally requires network access at construction (`hub.pull("rlm/rag-prompt")`) and
defines a `format_docs` it never uses.

**Resolved in PR-08, by keeping one of the three.**

- `rewriter.py` — **wired**. It runs only when retrieval scores below the
  relevance threshold, which is where it earns its model call: "how do i make my
  dude tougher" scored 0.041, and 0.439 after rewriting.
- `grader.py` — **deleted**. Wiring it as specced cost **31.7 s per query** and
  returned "yes" every time, because it re-evaluates the same ~1,000-token
  context the answer call is about to evaluate again. The retriever's own
  similarity score answers the same question for free. See #26.
- `generator.py` — **deleted**. Dead in every sense: no importers, network
  access required at construction, and `from langchain import hub` no longer
  imports on LangChain 1.x.

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

**Fixed across PR-09 and PR-10.**

- PR-00b untracked the PDFs; PR-09 rebuilt the index from the CC-BY SRD 5.1
  corpus in `corpus/srd/` (3,082 chunks, all `source = "SRD 5.1"`).
- PR-10 rewrote history with `git filter-repo`, dropping `Documents/*.pdf` and
  `chroma_db/` from all 35 commits, and gitignored both indexes as the build
  artifacts they are. **256 MB → <1 MB.** Verified: no blob matching
  `Documents/*.pdf` or `chroma_db` survives anywhere in `--all`.

**One caveat on the remote.** A force-push makes the old commits unreachable,
but GitHub does not garbage-collect them on a schedule you control — an object
is still fetchable by its exact SHA until GitHub runs GC. Ask GitHub Support to
run it if the exposure needs to be provably closed rather than merely
unreachable.

### 21. `env_activation.txt` is Windows-only

Contains `.\venv\Scripts\activate`. Fine as a personal note; misleading on this macOS
checkout, where the command is `source venv/bin/activate`.

**Deleted.** The README carries the correct command for the platform this runs
on. `conftest.py` went at the same time — it existed to put the repo root on
`sys.path`, which `pythonpath = ["."]` in `pyproject.toml` has done since PR-01.

## Found after the initial audit

### 22. `create_llm` hardcodes a model name that does not resolve

`src/models/llm.py` defaults to `model_name="Llama3.2"`. Verified against a live
daemon: `ResponseError: model 'Llama3.2' not found (status code: 404)`. Ollama
tags are lowercase and carry a size suffix (`llama3.2:3b`). Nothing in this repo
could ever have run against that default. → PR-02.

### 23. Chroma dirties the working tree on read

Opening the store modifies `chroma_db/chroma.sqlite3` — no writes required, the
open alone is enough. Since the store is committed, every app run and every test
collection leaves the repo dirty, and the mutation has had to be reverted out of
two PRs by hand. Either gitignore the store and make it a build artifact of
`scripts/ingest.py`, or move it out of the repo entirely. Unassigned.

### 24. Generation throughput is the bottleneck — and #6 wastes a whole generation

Measured on the target machine (Intel i9-9980HK, CPU-only — Ollama has no GPU
path on Intel Macs):

| | `llama3.2:3b` | `qwen2.5:7b` |
|---|---|---|
| Cold load (first call after start/eviction) | ~5 s | ~11 s |
| Warm routing call (system prompt + one turn) | **0.65 s** | — |
| Generation throughput | **11.4 tok/s** | **5.3 tok/s** |

**A correction.** This issue previously claimed routing cost 5.69 s per turn.
That number was a cold model load measured on a first call, not steady state.
Re-measured warm over six distinct routing prompts: mean **0.65 s**, and all six
routed correctly. Routing is neither slow nor inaccurate on this hardware.

What is slow is **generation**: `qwen2.5:7b` needs ~38 s for a 200-token answer.
That reframes #6. The cost of `dice_roller` returning to the supervisor is not
the extra 0.65 s routing call — it is that the supervisor then routes the roll
*result* to `researcher`, which spends **~40 s generating an answer to a question
nobody asked**. The wasted generation is ~60× the wasted routing call. → PR-04
(fix #6), PR-06 (stream, so generation is not experienced as a hang).

Both models stay resident together (2.6 GB + 5.1 GB, verified via `/api/ps`), so
the per-agent model map costs no swap penalty on a 32 GB machine.

### 25. A 3B model is not accurate enough to route

PR-02 put `llama3.2:3b` on the supervisor because routing is short and latency
was assumed to dominate. PR-04 measured it on a 12-case set spanning all four
destinations:

| Router model | Correct | Mean latency |
|---|---|---|
| `llama3.2:3b` | **7/12** | ~1.7 s |
| `qwen2.5:7b` | **12/12** | ~2.7 s |

Both prompts were tried on the 3B (the original terse one scored 8/12, a rewrite
with worked examples scored 7/12) — prompt wording was not the lever, model size
was. An earlier benchmark reporting 6/6 used cases with obvious keyword signals
and was too easy to be informative.

A routing decision is ~10 output tokens, so the larger model costs about **1 s**
more per turn. A misroute costs ~40 s of unwanted generation, or total silence
when it lands on `dungeon_master` while that agent is still a stub. Accuracy
dominates. `AGENT_MODELS["supervisor"]` is now `qwen2.5:7b`; set
`DND_MODEL_SUPERVISOR=llama3.2:3b` to trade it back. **Fixed** (PR-04).

### 26. Time-to-first-token is dominated by prompt evaluation

Once narration streams, the number a player feels is not throughput but how long
the screen stays empty. On CPU that is prompt-eval, and it scales with how much
history goes into the call:

| DM context window | Prompt size | Time to first token |
|---|---|---|
| 8 messages | ~478 tok | ~6.6 s |
| 6 messages | ~413 tok | ~5.2 s |
| 4 messages | ~348 tok | ~3.6 s |
| 2 messages | ~283 tok | ~2.2 s |

`CONTEXT_WINDOW` is 4. Continuity survives the cut because the durable facts are
extracted into `game_state` and replayed as a one-line briefing, so the model
gets the *state* of the world without re-reading the transcript of it.

Beware measuring this with a repeated identical prompt — Ollama's prompt cache
returns ~0.4 s and the number is meaningless. The same trap produced the wrong
routing figure in #24.

Not fully closed: the researcher does not stream (its RAG chain is PR-08's), and
its first token was measured at **61.8 s** on a cold embedding model.

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

### 27. A local model invents dice modifiers

Measured while replacing the dice parser with structured output. Asked to
extract a modifier, `llama3.2:3b` returned numbers that were nowhere in the
request:

| Request | Modifier returned | Correct |
|---|---|---|
| `roll 2d8 + 1d6 for damage` | **+1** | 0 |
| `roll a d20 with advantage for stealth` | **+5** | 0 |
| `roll 1d20 with disadvantage` | **-1** | 0 |
| `roll for initiative` | **+2** | 0 |

4/7 correct. A sharper prompt got it to 6/7; `qwen2.5:7b` got 7/7 but costs
**9.5 s** against 4.6 s. Every wrong answer silently changes the number the
player gets, which is worse than a visible failure.

**Fixed in PR-05, by not asking.** A dice expression is a formal language, so it
is read with a regex: notation, modifier, advantage, and stated purpose all come
straight from the request. The model is consulted only when the request names no
dice at all (`"roll for initiative"`), and even then the modifier is taken from
the player's words — because a number the request does not contain is a number
the model made up. That last guard was added after `1d20+2` came back for
`"roll for initiative"`: the invented bonus had been folded into the *notation*
string, sliding past a check that only looked at the modifier field.

The common path now costs **no LLM call at all** — 0.0 s, down from 4.6 s.

### 28. Cited page numbers are PDF pages, not printed pages

`src/data/loader.py` copies PyMuPDF's `page` into `page_number`, which is the
0-based index of the page *in the file*. A rulebook PDF includes a cover and
front matter, so this runs several pages behind the number printed on the page —
the Sneak Attack passage is cited as `Player's Handbook, p.89`, and the printed
page in the book is in the mid-90s.

This matters more now that PR-08 surfaces citations to the player, since the
entire point of a citation is that it can be checked. Two ways to fix it:

- Add a per-book offset to `DOCUMENT_PATHS` and apply it during ingestion. Cheap,
  but hand-measured per book and wrong if the PDF edition differs.
- Read the printed folio off the page text during ingestion. More robust, and
  these scans are OCR of varying quality.

Either requires re-indexing, so it belongs with the next corpus change. Until
then the labels are internally consistent — they do identify the retrieved
passage — just offset from the printed number.

**Moot on the default corpus since PR-09.** SRD chunks cite by entry name —
`SRD 5.1, Monsters: Goblin` — which is what a page number was a proxy for, and
is directly checkable. The issue still applies to `chroma_db_full/`, built from
the PDFs.
