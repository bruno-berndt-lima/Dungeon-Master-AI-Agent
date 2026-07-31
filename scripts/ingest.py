#!/usr/bin/env python
"""Build the ChromaDB index from the source rulebooks.

The three pieces this wires together — `load_documents`, `split_documents`,
`get_vectorstore` — were complete and correct from the first commit, and
imported by nothing. The committed `chroma_db/` was built out of band, so until
now the index could not be reproduced from the repository (KNOWN_ISSUES #12).

    python scripts/ingest.py              # build, refusing to touch an existing index
    python scripts/ingest.py --rebuild    # replace an existing index
    python scripts/ingest.py --dry-run    # load and chunk, but do not embed

The PDFs are gitignored (they are commercial Wizards of the Coast books), so a
fresh clone will not have them. `Documents/README.md` lists the filenames
`src/config.py` expects, and notes the CC-BY SRD 5.1 as a reproducible
alternative. You do not need any of this to *run* the app — `chroma_db/` is
committed.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow `python scripts/ingest.py` from the repo root without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CHROMA_DB_DIRECTORY, DOCUMENT_PATHS, EMBEDDING_MODEL_NAME
from src.data.loader import load_documents
from src.data.processing import CHUNK_OVERLAP, CHUNK_SIZE, split_documents
from src.data.vectorstore import build_vectorstore


def find_missing_documents(document_paths: dict) -> list:
    """Which configured PDFs are not on disk."""
    return [
        (name, path)
        for name, path in document_paths.items()
        if not Path(path).is_file()
    ]


def report_missing(missing: list) -> None:
    print("Cannot build the index — these source documents are missing:\n", file=sys.stderr)
    for name, path in missing:
        print(f"  {name:26} expected at {path}", file=sys.stderr)
    print(
        "\nThey are gitignored, so a fresh clone will not have them. See "
        "Documents/README.md for the expected filenames and for the SRD 5.1 "
        "alternative.\n"
        "\nYou do not need them to run the app: chroma_db/ is committed and "
        "retrieval works without them.",
        file=sys.stderr,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the ChromaDB index from the source rulebooks."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="replace an existing index. Without this, an existing index is left "
             "alone — Chroma appends, which would duplicate every chunk.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="load and chunk the documents, then stop. Reports the chunk count "
             "without paying for embeddings.",
    )
    parser.add_argument(
        "--persist-directory",
        default=CHROMA_DB_DIRECTORY,
        help=f"where to write the index (default: {CHROMA_DB_DIRECTORY})",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=CHUNK_SIZE,
        help=f"characters per chunk (default: {CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=CHUNK_OVERLAP,
        help=f"characters shared between neighbours (default: {CHUNK_OVERLAP})",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    missing = find_missing_documents(DOCUMENT_PATHS)
    if missing:
        report_missing(missing)
        return 1

    index_path = Path(args.persist_directory)
    if index_path.exists() and not args.rebuild and not args.dry_run:
        print(
            f"An index already exists at {args.persist_directory}. Pass --rebuild "
            f"to replace it.",
            file=sys.stderr,
        )
        return 1

    started = time.perf_counter()

    print(f"Loading {len(DOCUMENT_PATHS)} documents...")
    docs = load_documents(DOCUMENT_PATHS)
    print(f"  {len(docs)} pages")

    print(f"Chunking at {args.chunk_size} chars, {args.chunk_overlap} overlap...")
    chunks = split_documents(docs, args.chunk_size, args.chunk_overlap)
    print(f"  {len(chunks)} chunks")

    if args.dry_run:
        print(f"\nDry run — nothing written. {time.perf_counter() - started:.1f}s")
        return 0

    print(f"Embedding with {EMBEDDING_MODEL_NAME} (this is the slow part)...")
    store = build_vectorstore(chunks, args.persist_directory, rebuild=args.rebuild)

    indexed = store._collection.count()
    print(
        f"\nIndexed {indexed} chunks into {args.persist_directory} "
        f"in {time.perf_counter() - started:.1f}s"
    )
    if indexed != len(chunks):
        print(
            f"Warning: {len(chunks)} chunks were produced but {indexed} are in the "
            f"store.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
