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

**There is no ingestion entry point.** `loader.py` and `processing.py` are complete
and correct but never imported outside themselves; the committed `chroma_db/` was
built out-of-band. To rebuild:

```python
from src.config import DOCUMENT_PATHS
from src.data.loader import load_documents
from src.data.processing import split_documents
from src.data.vectorstore import get_vectorstore

docs   = load_documents(DOCUMENT_PATHS)   # PyMuPDFLoader per book
chunks = split_documents(docs)            # chunk_size=1000, chunk_overlap=200
store  = get_vectorstore(chunks)
```

**`src/data/loader.py`** iterates `DOCUMENT_PATHS` (a `{book_name: path}` dict), loads
each PDF with `PyMuPDFLoader`, and stamps every page's metadata with
`{"book": <name>, "page_number": doc.metadata["page"]}`. That metadata is the
foundation for the page citations `RESEARCHER_PROMPT` asks for — but nothing downstream
reads it. Surfacing `book` and `page_number` in the prompt context is the single
highest-value retrieval improvement available.

**`src/data/processing.py`** — `RecursiveCharacterTextSplitter(chunk_size=1000,
chunk_overlap=200)`. Fixed-size character splitting on rulebook PDFs breaks stat blocks
and spell descriptions mid-entry; 4,778 chunks for three books is coarse.

**`src/data/vectorstore.py`** — `get_vectorstore(docs)`:

```python
if os.path.exists(CHROMA_DB_DIRECTORY):
    return Chroma(persist_directory=..., embedding_function=embedding_model)
return Chroma.from_documents(documents=docs, embedding=..., persist_directory=...)
```

Two things to know: the `docs` argument is **silently ignored** when the directory
exists (delete `chroma_db/` to force a rebuild — there is no incremental add path),
and the module imports `Chroma` twice, from `langchain_community.vectorstores` and
then from `langchain_chroma`, with the second shadowing the first. Drop the
`langchain_community` import.

`ResearcherAgent` calls `get_vectorstore([])` with an empty list purely to hit the
load branch — a `load_vectorstore()` / `build_vectorstore(docs)` split would say what
it means.

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
