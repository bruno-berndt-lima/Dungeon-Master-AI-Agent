# Corpus

## `srd/` — System Reference Document 5.1

The default corpus. Committed, so `python scripts/ingest.py` works on a fresh
clone with no downloads and no PDFs.

### Attribution

This work includes material taken from the System Reference Document 5.1
("SRD 5.1") by Wizards of the Coast LLC, available at
<https://dnd.wizards.com/resources/systems-reference-document>. The SRD 5.1 is
licensed under the Creative Commons Attribution 4.0 International License,
available at <https://creativecommons.org/licenses/by/4.0/legalcode>.

The JSON transcription is from [5e-bits/5e-database](https://github.com/5e-bits/5e-database)
(MIT licensed for the compilation; the underlying game content remains under
CC-BY-4.0 as above).

### What is in it

21 files, ~2,300 entries, indexing to **3,082 chunks**.

| | entries | | entries |
|---|---|---|---|
| Monsters | 334 | Skills | 18 |
| Magic Items | 362 | Conditions | 15 |
| Spells | 319 | Damage Types | 13 |
| Class Features | 407 | Classes / Subclasses | 12 / 12 |
| Class Levels | 290 | Weapon Properties | 11 |
| Equipment | 237 | Races | 9 |
| Rule Sections | 33 + 6 | Alignments | 9 |
| Racial Traits | 38 | Magic Schools | 8 |
| | | Ability Scores | 6 |
| | | Languages | 16 |
| | | Backgrounds | 1 |

Deliberately excluded: `Equipment-Categories` and `Proficiencies`, which are
mostly cross-reference lists whose prose adds nothing anyone asks about.

### What is *not* in it

The SRD is a subset of the published rulebooks, and the gaps are real:

| | SRD | Published |
|---|---|---|
| Subclasses | 12 (one per class) | ~40 |
| Races | 9 | ~30 |
| Backgrounds | 1 (Acolyte) | ~13 |
| Feats | 1 (Grappler) | ~40 |
| Monsters | 334 | ~760 |

There is also almost no Dungeon Master's Guide material — no magic item creation,
encounter design tables, or campaign guidance.

If you own the books and want that coverage, build the second index; see
`Documents/README.md`. It is gitignored and stays on your machine.

### Why JSON rather than the SRD PDF

One document per *entry* instead of one per *page*:

- **Citations are names**, not page numbers. `SRD 5.1, Monsters: Goblin` can be
  looked up; a PDF page index is offset from the printed page (KNOWN_ISSUES #28).
- **No chunk boundary through a stat block.** A page-based split cut monsters in
  half at arbitrary points.
- **Clean text.** The scanned rulebooks were OCR'd, so dice notation came through
  as `ld6` and `ldl2` — an `l` for a `1`, in exactly the tokens a rules assistant
  exists to get right.
- **Real metadata** per chunk: `category`, `name`, and `cr` / `level` / `school`.
