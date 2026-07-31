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

Then build the second index:

```bash
python scripts/ingest.py --source rulebooks --rebuild
```

That writes to `chroma_db_full/`, leaving the default `chroma_db/` SRD-only.
Both are gitignored build artifacts. Point the app at it with:

```bash
DND_CHROMA_DIR=chroma_db_full python main.py
```

## You probably don't need them

`python scripts/ingest.py` builds the default index from the **SRD 5.1** corpus
in `corpus/srd/` in about 35 seconds — 3,082 chunks, 384-dim
`all-MiniLM-L6-v2` — using nothing but the repository.

These PDFs buy coverage the SRD does not have: ~40 subclasses instead of 12, ~30
races instead of 9, ~760 monsters instead of 334, and the DMG's guidance. See
`corpus/README.md` for the full comparison.
