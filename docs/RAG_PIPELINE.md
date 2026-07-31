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

`scripts/ingest.py` (PR-07) is the entry point. Before it, `loader.py` and
`processing.py` were complete, correct, and imported by nothing — the committed
`chroma_db/` was built out of band and could not be reproduced from the repo.

```bash
python scripts/ingest.py              # build; refuses to touch an existing index
python scripts/ingest.py --rebuild    # replace an existing index
python scripts/ingest.py --dry-run    # load and chunk, skip embedding
```

It also takes `--persist-directory`, `--chunk-size` and `--chunk-overlap`, so an
experiment can be built somewhere else without disturbing the committed store.

**The PDFs are gitignored**, so a fresh clone has none of them and the script
says so by name and expected path rather than failing on a stack trace. You do
not need them to *run* the app.

**`src/data/loader.py`** iterates `DOCUMENT_PATHS` (a `{book_name: path}` dict), loads
each PDF with `PyMuPDFLoader`, and stamps every page's metadata with
`{"book": <name>, "page_number": doc.metadata["page"]}`. That metadata is the
foundation for the page citations `RESEARCHER_PROMPT` asks for — but nothing downstream
reads it. Surfacing `book` and `page_number` in the prompt context is the single
highest-value retrieval improvement available (PR-08).

**`src/data/processing.py`** — `RecursiveCharacterTextSplitter(chunk_size=1000,
chunk_overlap=200)`, now imported from `langchain_text_splitters`; the old
`langchain.text_splitter` path was removed in LangChain 1.x, and because nothing
imported the module the broken import never failed a test run. Fixed-size
character splitting on rulebook PDFs breaks stat blocks and spell descriptions
mid-entry; 4,778 chunks for three books is coarse. The defaults are module
constants now — changing either invalidates the committed index.

**`src/data/vectorstore.py`** — split into two functions that say what they do:

```python
load_vectorstore()                       # read-only; raises if no index exists
build_vectorstore(docs, rebuild=False)   # writes; refuses to append silently
```

`get_vectorstore(docs)` used to be both, and **silently discarded `docs`** whenever
the directory existed — which is why `ResearcherAgent` called it with `[]` to hit
the load branch. It now calls `load_vectorstore()`. Two failure modes that used
to be invisible now raise:

- **No index on disk.** The old code returned an empty but usable store, so a
  missing index looked like a working one that retrieved nothing.
- **Building over an existing index.** Chroma appends — verified, 6 chunks
  became 7 — so this would have silently duplicated every chunk. `--rebuild`
  deletes first.

The duplicate `Chroma` import (`langchain_community.vectorstores` then
`langchain_chroma`, second shadowing the first) is gone.

## The unused corrective-RAG layer

`src/pipelines/` contains three factories that together form a standard CRAG loop.
None are imported anywhere.

**`grader.py`** — `create_retrieval_grader(llm)` returns
`grade_prompt | llm.with_structured_output(GradeDocuments)`, where `GradeDocuments` is
a `pydantic.v1.BaseModel` with a single `binary_score: str` field ('yes'/'no').
Note two dependencies: `pydantic.v1` (legacy shim), and `with_structured_output`,
which `ChatOllama` supports unevenly depending on the model.

**`rewriter.py`** — `create_question_rewriter(llm)`, a D&D-aware query rewriter that
takes the original question and produces a version better aligned to 5e terminology
and mechanics. This is the piece most likely to pay off immediately: player phrasing
("can I sneak past the guard?") and rulebook phrasing ("Stealth check, Passive
Perception") are very far apart in embedding space.

**`generator.py`** — `create_rag_chain(llm)` pulls `rlm/rag-prompt` from LangChain Hub
and pipes it through the LLM. It requires network access at construction time and
duplicates what `ResearcherAgent` already builds inline. It also defines a local
`format_docs` that it never wires into the chain.

Wiring these in would give: retrieve → grade each doc → if none relevant, rewrite the
query and retry → generate. That is the design the file layout implies but the code
never assembled.

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
the embedder means re-indexing all three books from scratch. Swapping the *chat* model
(`src/models/llm.py`) touches nothing in the index.

## Licensing note

The PDFs are gitignored as of PR-00b, but two exposures remain on this **public** repo:

1. **The index contains the books' text verbatim.** Chroma stores each chunk's
   `page_content` alongside its vector — `select string_value from embedding_metadata
   where key='chroma:document'` returns 4,778 rows of readable rulebook prose. Removing
   the PDFs did not remove the text.
2. **Git history still has the PDFs.** They were added in the initial commit
   (`d617836`) and remain fetchable by SHA until history is rewritten.

The SRD 5.1 is published under CC-BY-4.0 and covers core mechanics, conditions, most
spells, and a large monster set. Re-indexing from the SRD would clear both exposures
and make ingestion reproducible for anyone cloning the repo, at the cost of
DMG-specific guidance and non-SRD monsters.
