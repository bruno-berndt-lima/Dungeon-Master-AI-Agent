# RAG pipeline

## What's in the index

| Property | Value |
|---|---|
| Store | ChromaDB, persisted at `chroma_db/` (committed to git) |
| Collection | `langchain` |
| Vectors | 4,778 |
| Dimensions | 384 |
| Distance | L2 |
| Index | HNSW, `M=16`, `ef_construction=100`, `ef_search=100` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` via `HuggingFaceEmbeddings` |

Sources are the three core 5e books: Player's Handbook, Dungeon Master's Guide,
Monster Manual. The PDFs themselves are **gitignored** and not distributed with the
repo (`Documents/README.md`); the built index is committed, so retrieval works without
them.

## Active path

Only one consumer exists — `ResearcherAgent`:

```
query ──▶ vectorstore.as_retriever()      # similarity, k=4 (defaults)
      ──▶ Document[] interpolated into {context}
      ──▶ ChatPromptTemplate(RESEARCHER_PROMPT + context, user question)
      ──▶ ChatOllama(Llama3.2, temperature=0)
      ──▶ StrOutputParser()
```

Built once in `ResearcherAgent.__init__` as an LCEL chain. If construction throws
(missing store, missing model download, etc.) it is caught, a warning is printed, and
`rag_chain` is set to `None` — after which `process_task` answers from the bare LLM
with no retrieval at all. The `rag_used` boolean in each log entry's metadata tells
you which path ran.

## Ingestion

Two corpora, two loaders, two indexes.

```bash
python scripts/ingest.py --rebuild                       # SRD 5.1  -> chroma_db/
python scripts/ingest.py --source rulebooks --rebuild    # your PDFs -> chroma_db_full/
python scripts/ingest.py --dry-run                       # chunk without embedding
```

`chroma_db/` is committed and built from `corpus/srd/`, which also ships with the
repository — so the index is reproducible from a clean clone with no PDFs and no
network. `chroma_db_full/` is gitignored. `DND_CHROMA_DIR` selects which one the
app reads.

### `src/data/srd_loader.py` — the default path

One document per **entry**, not per page. A monster, a spell, a rule section.

```
# Goblin
Small humanoid (goblinoid), neutral evil
Armor Class 15 (Leather Armor, Shield)
Hit Points 7 (2d6)
...
Actions
Scimitar. Melee Weapon Attack: +4 to hit, reach 5 ft., one target.
Hit: 5 (1d6 + 2) slashing damage.

metadata: source="SRD 5.1" category="Monsters" name="Goblin" cr="0.25" type="humanoid"
```

Three things it does that a blind split cannot:

- **Re-heads every chunk with the entry name.** Chroma embeds `page_content` and
  never metadata, so a rule section split into four pieces would leave three that
  cannot be found by name.
- **Keeps a stat block whole** where it fits, so a creature's actions never land
  in a different chunk from its hit points.
- **Merges entries that are the same text on different classes.** `Ability Score
  Improvement` is stored once per class per level — 50 identical copies. Indexed
  separately they crowded unrelated results: a query about a goblin's armour
  class returned three copies of `Fighting Style: Defense` in the top six, and
  the Goblin only at rank 2. Merged, the Goblin ranks first.

Measured effect of the corpus change on retrieval:

| Query | Rulebook index | SRD index |
|---|---|---|
| "how does sneak attack work for rogues" | 0.428 — *a rogue intro page* | **0.590 — the Sneak Attack feature** |
| "how much damage does a fireball do" | 0.363 — *a monster's spell list* | **0.513 — the Fireball spell** |
| "how does grappling work in combat" | 0.529 — *the Grappler feat* | 0.335 — **the grappling rules** |

Scores are not comparable across indexes, but the retrieved *content* is: the SRD
index lands on the entry the question was about.

### `src/data/loader.py` — the PDF path

Iterates `DOCUMENT_PATHS`, loads each PDF with `PyMuPDFLoader`, and stamps every
page with `{"book": <name>, "page_number": doc.metadata["page"]}`. Still
supported, still the way to index the full rulebooks — with two caveats recorded
in KNOWN_ISSUES: the page numbers are PDF indices rather than printed pages
(#28), and the scans are OCR, so dice notation arrives as `ld6` and `ldl2`.

### `src/data/processing.py`

`RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`, imported
from `langchain_text_splitters`; the old `langchain.text_splitter` path was
removed in LangChain 1.x, and because nothing imported the module the broken
import never failed a test run. `split_documents` splits `Document`s for the PDF
path; `split_text` splits raw strings for the SRD loader, which re-heads the
pieces itself.

### `src/data/vectorstore.py`

```python
load_vectorstore()                       # read-only; raises if no index exists
build_vectorstore(docs, rebuild=False)   # writes; refuses to append silently
```

`get_vectorstore(docs)` used to be both, and **silently discarded `docs`** whenever
the directory existed — which is why `ResearcherAgent` called it with `[]`. It now
calls `load_vectorstore()`. Two failure modes that used to be invisible now raise:

- **No index on disk.** The old code returned an empty but usable store, so a
  missing index looked like a working one that retrieved nothing.
- **Building over an existing index.** Chroma appends — verified, 6 chunks
  became 7 — so this would have silently duplicated every chunk. `--rebuild`
  deletes first.

## Corrective RAG

`src/pipelines/` held three chains written in early 2025 and wired into nothing.
PR-08 kept one:

- **`rewriter.py` — wired.** Called only when the first retrieval scores below
  `RELEVANCE_THRESHOLD`, which is where a model call is worth its ~4 s. Measured:
  "how do i make my dude tougher" scored 0.041, and 0.439 after rewriting to
  "How can you increase your character's Armor Class, Hit Points, and
  Constitution?". Its prompt had to be tightened to return a bare question — the
  original asked the model to "formulate an improved question" and got
  commentary with it, which was then embedded along with the question.
- **`grader.py` — deleted.** Wiring it as specced cost **31.7 s per query** and
  answered "yes" every time. It re-evaluates the same ~1,000-token context the
  answer call is about to evaluate again, which on CPU is the single most
  expensive thing available. The retriever's own similarity score decides the
  same question for free.
- **`generator.py` — deleted.** No importers, `hub.pull("rlm/rag-prompt")`
  needed network access at construction, and `from langchain import hub` no
  longer imports on LangChain 1.x.

The relevance gate, measured over this index:

| | score range |
|---|---|
| On-topic rules questions | 0.363 – 0.529 |
| Off-topic questions | -0.154 – 0.053 |

`RELEVANCE_THRESHOLD = 0.25` sits in the gap. It is a property of *this* index
and embedding model — re-measure after any corpus change.

Per-*chunk* scores do not discriminate as cleanly: for "sneak attack" the four
retrieved chunks scored 0.428, 0.369, 0.349 and 0.301, and the 0.301 one was an
unrelated Monster Manual page. Pruning on that signal would risk dropping useful
context, so all `k` chunks are kept and the citation list says "passages
consulted" rather than claiming each one was used.

## Where to improve, in order of payoff

1. **Cite sources.** The `book` / `page_number` metadata is already on every chunk.
   Format it into `{context}` and the researcher can actually do what its prompt claims.
2. **Rewrite queries.** `rewriter.py` exists; put it in front of the retriever.
3. **Tune retrieval.** `as_retriever()` takes no arguments today. Try
   `search_type="mmr"` and `k=6–8`; four 1000-char chunks is thin for a rules question.
4. **Chunk semantically.** Split on rulebook structure (spell entries, stat blocks,
   section headers) rather than character count.
5. **Grade and retry.** `grader.py` closes the loop, but it needs a model with reliable
   structured output — this becomes straightforward after the provider swap described
   in `docs/REFACTOR_NOTES.md`.

## Embeddings and the provider swap

The embedding model runs locally through `sentence-transformers` and is independent of
whichever chat model you use. Keep it that way during any refactor: the committed
`chroma_db/` is built from 384-dimensional `all-MiniLM-L6-v2` vectors, and changing
the embedder means re-indexing the whole corpus from scratch. Swapping the *chat* model
(`src/models/llm.py`) touches nothing in the index.

## Licensing

The default corpus is the **SRD 5.1**, published by Wizards of the Coast under
**CC-BY-4.0**. It is vendored in `corpus/srd/` and the attribution notice CC-BY
requires is in `corpus/README.md`. Both the corpus and the index it produces are
redistributable.

**What PR-09 cleared.** The committed index used to hold the three rulebooks'
text verbatim — Chroma stores each chunk's `page_content` next to its vector, so
`select string_value from embedding_metadata where key='chroma:document'`
returned 4,778 rows of readable commercial prose. Rebuilt from the SRD, the same
query returns 3,082 rows, every one of them CC-BY. Verified: all chunks carry
`source = "SRD 5.1"`, and none match the old OCR artifacts.

**What is still open: git history.** The PDFs were added in the initial commit
(`d617836`), and the rulebook-derived index is in every commit up to PR-09. Both
stay fetchable by SHA until a `git filter-repo` rewrite plus force-push — a
destructive operation on a public repo, and a separate decision. See
KNOWN_ISSUES #20.

**Building from the rulebooks is still supported** (`--source rulebooks`), but it
writes to `chroma_db_full/`, which is gitignored. Wider coverage stays local.
