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

That writes to `chroma_db_full/`, which is **gitignored** — the committed
`chroma_db/` stays SRD-only. Point the app at it with:

```bash
DND_CHROMA_DIR=chroma_db_full python main.py
```

## You probably don't need them

`chroma_db/` is committed and built from the **SRD 5.1** corpus that ships in
`corpus/srd/` — 3,082 chunks, 384-dim `all-MiniLM-L6-v2`. Retrieval works out of
the box, and `python scripts/ingest.py --rebuild` reproduces it from the
repository alone.

These PDFs buy coverage the SRD does not have: ~40 subclasses instead of 12, ~30
races instead of 9, ~760 monsters instead of 334, and the DMG's guidance. See
`corpus/README.md` for the full comparison.
