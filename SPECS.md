# SPECS — Claude refactor

Execution contract for the refactor described in `docs/REFACTOR_NOTES.md`.
Each spec below is **one branch, one PR**. Read `CLAUDE.md` and
`docs/KNOWN_ISSUES.md` before starting any of them.

Issue numbers in `[#n]` refer to `docs/KNOWN_ISSUES.md`.

---

## How we work

**Branches** — `<type>/<scope>`, where type is one of `docs`, `chore`, `refactor`,
`feat`, `fix`. Branch off `main`, never off another feature branch.

**PRs** — one spec per PR. Every PR body must state: the spec ID, what changed, and
how it was verified. No PR merges without Bruno's approval.

**Scope discipline** — a PR touches only the files listed under *In scope*. If a
change appears to require editing a file outside that list, stop and raise it in the
PR rather than widening the diff.

**Merge policy** — squash merge, delete branch after. Rebase onto `main` before
requesting review if `main` has moved.

**Verification** — from PR-01 onward, `pytest` must pass before requesting review.
Paste the output in the PR body.

---

## PR map

| ID | Branch | Depends on | Parallel-safe with |
|---|---|---|---|
| PR-00 | `docs/refactor-baseline` | — | everything |
| PR-00b | `chore/untrack-wotc-pdfs` | — | everything |
| PR-01 | `chore/tooling-and-test-gate` | — | PR-00b, PR-07 |
| PR-02 | `refactor/claude-provider` | PR-01 | PR-03, PR-07 |
| PR-03 | `refactor/state-contract` | PR-01 | PR-02, PR-07 |
| PR-04 | `refactor/supervisor-structured-routing` | PR-02, PR-03 | PR-05, PR-06 |
| PR-05 | `refactor/dice-strict-tool` | PR-02, PR-03 | PR-04, PR-06 |
| PR-06 | `feat/dungeon-master` | PR-02, PR-03 | PR-04, PR-05 |
| PR-07 | `feat/ingest-script` | — | everything |
| PR-08 | `feat/researcher-citations-and-crag` | PR-02, PR-03 | — |

Three waves: **{PR-01, PR-07}** → **{PR-02, PR-03}** → **{PR-04, PR-05, PR-06}** → **PR-08**.

PR-04 and PR-06 both edit `src/prompts/prompts.py`, but different constants separated
by other text — git merges these cleanly. Rebase before review anyway.

---

## PR-00 — Documentation baseline

**Branch** `docs/refactor-baseline`

**In scope** `CLAUDE.md`, `SPECS.md`, `docs/*.md`

**Task** Land the repo analysis and this spec so every later PR (and any subagent) has
a shared, in-repo source of truth.

**Acceptance**
- `CLAUDE.md` at root; `ARCHITECTURE.md`, `AGENTS.md`, `RAG_PIPELINE.md`,
  `KNOWN_ISSUES.md`, `REFACTOR_NOTES.md` under `docs/`
- No source files touched

---

## PR-00b — Untrack the source rulebooks

**Branch** `chore/untrack-wotc-pdfs`

**In scope** `.gitignore`, `Documents/README.md` (new), `CLAUDE.md`,
`docs/RAG_PIPELINE.md`, `SPECS.md`

**Task** Stop tracking the three commercial WotC PDFs (128 MB) on a public repo.
`chroma_db/` stays committed so the app runs without them.

1. `git rm --cached Documents/*.pdf` — untrack, **keep local copies**; they are needed
   to re-index in PR-07.
2. Add `Documents/*.pdf` to `.gitignore`.
3. Add `Documents/README.md` documenting the expected filenames and the SRD 5.1
   alternative.
4. Correct the docs that describe the PDFs as committed.

**Acceptance**
- `git ls-files Documents/` lists only `README.md`
- The PDFs still exist on disk
- `python main.py` still retrieves — the app does not depend on the PDFs

**Out of scope — deliberately** Two exposures survive this PR and need separate
decisions:
- **History.** The PDFs were added in the initial commit (`d617836`) and stay
  fetchable by SHA until a `git filter-repo` rewrite plus force-push.
- **The index itself.** Chroma stores each chunk's text verbatim —
  `select string_value from embedding_metadata where key='chroma:document'` returns
  4,778 rows of readable rulebook prose. Only re-indexing from the SRD clears this.

---

## PR-01 — Tooling and test gate

**Branch** `chore/tooling-and-test-gate`

**In scope** `pyproject.toml`, `requirements.txt`, `tests/`, `conftest.py`

**Task**
1. `requires-python = ">=3.11"` — `src/agents/supervisor.py` uses `Literal[*...]`
   (PEP 646), which does not parse on 3.9 `[#2]`.
2. Fix the pytest table: `[tool.pytest]` → `[tool.pytest.ini_options]` `[#18]`.
3. Pin every dependency to a known-good version. Remove the duplicate `langchain`
   entry and drop `tavily-python` (never imported) `[#19]`.
4. Write `tests/test_dice.py` — real pytest coverage for `src/utils/dice.py`:
   `parse_dice_string` on `1d20`, `2d6+1d8`, bare `d20`; `roll_single_type` bounds;
   `roll_multiple` totals. Seed `random` for determinism.
5. Write `tests/test_graph_smoke.py` — compile the graph and assert the node set is
   `{supervisor, dungeon_master, researcher, dice_roller}`. If construction requires a
   live model or the Chroma index, mark it `@pytest.mark.integration` and keep the
   default run offline.
6. Delete or convert the two existing print-based scripts — `test_dice_roller.py`
   asserts nothing and cannot fail `[#17]`.

**Acceptance**
- `pytest` runs green offline, with no Ollama daemon
- At least one test genuinely fails if `DiceRoller.parse_dice_string` is broken

**Out of scope** Fixing the `-` separator bug in `parse_dice_string` `[#9]` — write a
test marked `xfail` for it; the fix belongs in PR-05.

---

## PR-02 — Claude provider

**Branch** `refactor/claude-provider`

**In scope** `src/models/llm.py`, `requirements.txt`, plus the single
`create_llm(...)` call line in each of the four agent `__init__` methods

**Task**
1. Replace `ChatOllama` with Claude. Add `anthropic` (and `langchain-anthropic` if
   staying on LCEL) to requirements.
2. **Remove the `temperature` parameter entirely.** `temperature`, `top_p`, and
   `top_k` return 400 on Fable 5 / Opus 5 and on non-default values for Sonnet 5.
   The current default is `temperature=0`, so this will fail immediately if carried over.
3. Per-agent model map:
   | Agent | Model |
   |---|---|
   | `dungeon_master` | `claude-fable-5` |
   | `researcher` | `claude-opus-5` |
   | `supervisor` | `claude-haiku-4-5` |
   | `dice_roller` | `claude-haiku-4-5` |
   `create_llm(agent_type)` resolves the model from this map.
4. Do **not** send a `thinking` parameter on `claude-fable-5` — thinking is always on
   there and an explicit `{"type": "disabled"}` returns 400. Control depth with
   `output_config: {"effort": ...}`. Use `low` or `medium` on the interactive path.
5. Handle `stop_reason == "refusal"` before reading response content.
6. Delete `create_json_llm` — unused, and superseded by structured output `[#15]`.

**Acceptance**
- `grep -ri "ollama\|llama3" src/` returns nothing
- No `temperature`/`top_p`/`top_k` anywhere in `src/`
- One real call per agent role succeeds against the API
- `pytest` still green

**Out of scope** Embeddings. `all-MiniLM-L6-v2` stays — the committed `chroma_db/` is
built from its 384-dim vectors and swapping it invalidates the index.

---

## PR-03 — State contract

**Branch** `refactor/state-contract`

**In scope** `src/graph/game_state.py`, `src/graph/game_orchestrator.py`, `main.py`,
`src/agents/base_agent.py`, `src/agents/supervisor.py`, `src/agents/researcher.py`,
`src/agents/dice_roller.py`, `requirements.txt`, `tests/`

> **Scope widened during execution (approved).** Removing the `next_agent` field
> means touching every writer — `supervisor.py` and `researcher.py` both set it,
> and `main.py` read it. `requirements.txt` gains
> `langgraph-checkpoint-sqlite` for the checkpointer.
>
> A correction to task 1 below: `add_messages` merges on **message id**, not
> position, and messages read out of state already carry ids. The pre-existing
> whole-list return therefore did *not* corrupt history. Returning deltas is
> still preferred — smaller payloads, idiomatic — but it is a style improvement,
> not the correctness fix this spec originally implied. The real hazard is
> rebuilding message objects from scratch, which drops their ids and does
> duplicate; `tests/test_state_contract.py` pins both behaviors.

**Task**
1. `messages: Annotated[Sequence[BaseMessage], add_messages]` — removes the
   copy-the-whole-list pattern and the silent-history-drop hazard `[#7]`.
2. Commit to `BaseMessage` as the single message representation. Drop the dual-shape
   handling in `BaseAgent._get_latest_message` only if every producer is converted in
   this PR; otherwise leave it and note the follow-up.
3. Fix the `game_state` type collision — `main.py` overwrites the factory's `{}` with
   the string `"initialized"`, which silently turns
   `BaseAgent.initialize_agent`'s key lookup into a substring check `[#4]`.
4. Add a `SqliteSaver` checkpointer so campaign state survives across sessions.
5. Drop the `state["next_agent"]` mirror — `Command(goto=...)` is the real routing
   channel and the two can diverge `[#15, #16]`.
6. Fix `ResearcherAgent`'s return annotation: it says `Command[Literal["supervisor"]]`
   and returns `goto="__end__"` `[#3]`. Annotation only — no behavior change here.

**Acceptance**
- `pytest` green, graph smoke test still passes
- No key is written that is not declared in `GameState`
- A second run resumes prior state from the checkpoint

---

## PR-04 — Supervisor structured routing

**Branch** `refactor/supervisor-structured-routing`

**In scope** `src/agents/supervisor.py`, `SUPERVISOR_PROMPT` in `src/prompts/prompts.py`

**Task**
1. Replace the substring-match ladder with structured output over the `Router`
   TypedDict already declared (and unused) in the module `[#5, #15]`.
2. Delete the catch-all `goto = "researcher"` fallback. A parse failure should be an
   explicit error path, not silent misrouting.
3. Fix the post-dice loop: after `dice_roller` returns, the newest message is the roll
   result and the supervisor routes on *that*, sending it to `researcher` to answer an
   unasked question `[#6]`. Verified in
   `logs/llm_interactions/llm_log_2025-03-31.jsonl`. Either give the router an
   explicit "request already satisfied → FINISH" branch, or route on `current_task`
   rather than the full tail.
4. Delete the unused `class State(MessagesState)`.

**Acceptance**
- No substring matching on model output anywhere in the file
- A dice-only request produces exactly one dice roll and terminates — no researcher hop
- Routing decisions still logged via `_log_interaction`

---

## PR-05 — Dice roller strict tool

**Branch** `refactor/dice-strict-tool`

**In scope** `src/agents/dice_roller.py`, `src/utils/dice.py`, `tests/test_dice.py`

**Task**
1. Replace `_parse_dice_request` with a strict tool schema
   (`"strict": True`, `additionalProperties: false`). This deletes the markdown-fence
   regex, the `PQXYpqxy` placeholder detection, the string-modifier coercion, and the
   `for`/`to check` description salvage — all of it exists only because Llama 3.2
   could not emit reliable JSON.
2. Move the inline parse prompt into `src/prompts/prompts.py` as a constant, per the
   convention in `CLAUDE.md`.
3. Fix `DiceRoller.parse_dice_string` to handle `-` separators; flip the `xfail` test
   from PR-01 `[#9]`.
4. Fix the shared-state mutation: `dict(state)` is shallow, so
   `updated_state["messages"].append(...)` mutates the caller's list. Match
   `ResearcherAgent`'s `list(...)` copy `[#8]`.
5. Delete `class DiceRollRequest(Dict[str, Any])` — not a usable structure, never
   instantiated `[#15]`.

**Acceptance**
- `DiceRoller` still performs every roll; the model only parses
- `2d6-1`, `1d20+5`, `2d8 + 1d6`, advantage and disadvantage all correct
- `pytest` green with the previously-`xfail` test now passing

---

## PR-06 — Dungeon Master

**Branch** `feat/dungeon-master`

**In scope** `src/agents/dungeon_master.py`, `DUNGEON_MASTER_PROMPT` in
`src/prompts/prompts.py`

**Task** Implement `process_task`. It is currently `pass`, so the node returns `None`
instead of a `Command` — this is the project's core agent and the single largest gap
`[#1]`.

Everything it needs already exists: the prompt constant, the LLM factory, the state
schema, `_log_interaction`, and the `Command` return pattern. Follow `ResearcherAgent`
for structure.

1. Build narrative context from `state["messages"]` and `state["game_state"]`.
2. Call the LLM (`claude-fable-5` per PR-02), stream the response — this is the one
   place a player feels latency.
3. Append the narration as an assistant message.
4. Return `Command(goto=...)` with an annotation that matches what it actually returns.
5. Update `game_state` when the narration establishes new locations, items, or effects,
   as `DUNGEON_MASTER_PROMPT` already instructs.

**Acceptance**
- A narrative prompt routed to `dungeon_master` returns prose and does not error
- Return annotation matches the actual `goto`
- Interaction logged

---

## PR-07 — Ingest script

**Branch** `feat/ingest-script`

**In scope** `scripts/ingest.py` (new), `src/data/vectorstore.py`,
`src/data/processing.py`

> **Blocked on a 1.x import fix.** `src/data/processing.py` does
> `from langchain.text_splitter import RecursiveCharacterTextSplitter`, which no
> longer exists — the class moved to the standalone `langchain_text_splitters`
> package. Nothing imports the module today, so it does not fail the test suite,
> but `scripts/ingest.py` cannot work until it is fixed. Found while verifying
> PR-01 on Python 3.12.

**Task**
1. Write `scripts/ingest.py` wiring the three existing, correct, never-imported
   functions: `load_documents` → `split_documents` → `get_vectorstore` `[#12]`.
   The committed `chroma_db/` was built out-of-band and is currently not reproducible
   from the repo.
2. Split `get_vectorstore(docs)` into `load_vectorstore()` and
   `build_vectorstore(docs)`. Today it silently ignores its `docs` argument whenever
   `chroma_db/` exists, which is why `ResearcherAgent` calls it with `[]` `[#11]`.
3. Remove the duplicate `Chroma` import — the module imports it from
   `langchain_community.vectorstores` and then from `langchain_chroma` `[#10]`.
4. Add a `--rebuild` flag that clears the directory first.

**Acceptance**
- `python scripts/ingest.py --rebuild` reproduces an index of comparable size
  (~4,778 chunks, 384-dim) from the PDFs in `Documents/`
- The script fails with a clear message when the PDFs are absent — they are gitignored
  as of PR-00b, so a fresh clone will not have them
- `ResearcherAgent` no longer calls `get_vectorstore([])`

---

## PR-08 — Researcher citations and corrective RAG

**Branch** `feat/researcher-citations-and-crag`

**In scope** `src/agents/researcher.py`, `src/pipelines/*`

**Task**
1. **Citations.** `src/data/loader.py` already stamps every chunk with `book` and
   `page_number`, and nothing downstream reads it. Format both into `{context}` so the
   researcher can do what `RESEARCHER_PROMPT` already claims. Highest-value retrieval
   change available.
2. **Query rewriting.** Put `create_question_rewriter` in front of the retriever.
   Player phrasing and rulebook phrasing are far apart in embedding space `[#13]`.
3. **Grading + retry.** Wire `create_retrieval_grader`; on a no-relevant-docs result,
   rewrite and retry once. This was never wired in because it depends on
   `with_structured_output`, which is unreliable on Ollama and reliable on Claude.
4. Tune retrieval — `as_retriever()` currently takes no arguments (similarity, `k=4`).
   Try `search_type="mmr"`, `k=6–8`.
5. Delete `src/pipelines/generator.py` outright. It duplicates the inline chain,
   needs network access at construction to pull `rlm/rag-prompt`, and its
   `from langchain import hub` no longer imports on LangChain 1.x — `hub` was
   removed. Nothing imports the module, so it is dead in every sense `[#13, #15]`.
   Also delete or wire up `ResearcherAgent.format_docs`, currently unused.
6. Migrate `grader.py` off `pydantic.v1`.

**Acceptance**
- Rules answers cite book and page
- `metadata.rag_used` is `True` on the retrieval path
- A deliberately obscure query triggers one rewrite-and-retry, visible in the logs

---

## Deferred

Not scheduled — decide separately.

> **Licensing items: consciously deferred 2026-07-31.** The two rulebook-exposure
> items below were reviewed and left as-is; they are not oversights. Revisit before
> any of: promoting the repo, adding collaborators, accepting outside contributions,
> or re-indexing for another reason. The SRD 5.1 switch is cheapest to fold into PR-07
> while `scripts/ingest.py` is still unwritten — after that it costs a rewrite.

- **Agent registry consolidation.** Adding an agent means editing four places: node
  registration, `AGENT_TYPES`, the supervisor's parse chain, and `SUPERVISOR_PROMPT`.
- **`src/actors/` activation.** `Player` and `NPC` are type-hint-only today; they
  become useful once the checkpointer from PR-03 makes `players`/`npcs` worth
  populating `[#14]`.
- **History rewrite.** PR-00b untracked the PDFs but they remain in the initial commit
  and fetchable by SHA. Purging them needs `git filter-repo` plus a force-push, which
  rewrites every SHA in the repo `[#20]`.
- **Re-index from SRD 5.1.** `chroma_db/` holds the rulebooks' text verbatim (4,778
  chunks). Rebuilding the index from the CC-licensed SRD is the only thing that clears
  that, and it would make ingestion reproducible for anyone cloning the repo.
- **`env_activation.txt`** is Windows-only and misleading on macOS `[#21]`.
