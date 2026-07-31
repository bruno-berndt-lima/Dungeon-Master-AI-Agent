# Source documents

The PDFs this project indexes are **not distributed with the repository** — they are
commercial Wizards of the Coast rulebooks. `Documents/*.pdf` is gitignored.

To rebuild the vector index yourself, place your own copies here under the filenames
`src/config.py` expects:

| Key in `DOCUMENT_PATHS` | Expected filename |
|---|---|
| Dungeon Master's Guide | `Dungeon_Masters_Guide_5e.pdf` |
| Player's Handbook | `Players_Handbook_5e.pdf` |
| Monster Manual | `Monster_Manual_5e.pdf` |

Then run the ingest script (see `SPECS.md` → PR-07).

## You probably don't need them

`chroma_db/` is committed and already contains the built index (4,778 chunks,
384-dim `all-MiniLM-L6-v2`). Retrieval works out of the box without these PDFs.
You only need them to re-index from scratch — for example after changing the chunking
strategy or the embedding model.

## Reproducible alternative

The **SRD 5.1** is published by Wizards of the Coast under CC-BY-4.0 and covers most
of what the rules-lookup path actually needs: core mechanics, conditions, most spells,
and a large monster set. Switching the corpus to the SRD would make ingestion legally
reproducible for anyone cloning this repo, at the cost of DMG-specific guidance and
non-SRD monsters.
