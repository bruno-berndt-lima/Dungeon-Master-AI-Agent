"""Tests for the ingestion path: `scripts/ingest.py` and `src/data/`.

The guard clauses all fire *before* any embedding model is constructed, so these
run in milliseconds and need no model download. The one test that actually
embeds is marked `slow` and builds three tiny PDFs of its own.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

from src.data.processing import CHUNK_OVERLAP, CHUNK_SIZE, split_documents
from src.data.vectorstore import (
    VectorStoreMissingError,
    build_vectorstore,
    load_vectorstore,
)
from scripts.ingest import find_missing_documents

pytestmark = pytest.mark.integration


# --- the import that was broken ---------------------------------------------

def test_the_splitter_imports():
    """`langchain.text_splitter` was removed in LangChain 1.x. Nothing imported
    this module before PR-07, so the broken import never failed a test run."""
    from src.data.processing import RecursiveCharacterTextSplitter  # noqa: F401


def test_chunking_preserves_metadata():
    """Book and page have to survive the split — they are what citations use."""
    docs = [Document(page_content="rule text. " * 400,
                     metadata={"book": "Player's Handbook", "page_number": 87})]
    chunks = split_documents(docs)

    assert len(chunks) > 1
    assert all(c.metadata["book"] == "Player's Handbook" for c in chunks)
    assert all(c.metadata["page_number"] == 87 for c in chunks)


def test_chunk_defaults_match_the_committed_index():
    """Changing these silently invalidates chroma_db/."""
    assert (CHUNK_SIZE, CHUNK_OVERLAP) == (1000, 200)


def test_chunks_respect_the_size_limit():
    docs = [Document(page_content="word " * 2000, metadata={})]
    assert all(len(c.page_content) <= CHUNK_SIZE for c in split_documents(docs))


# --- missing source documents -----------------------------------------------

def test_missing_documents_are_reported_not_ignored(tmp_path):
    present = tmp_path / "here.pdf"
    present.write_bytes(b"%PDF-1.4")

    missing = find_missing_documents({
        "Present": str(present),
        "Absent": str(tmp_path / "gone.pdf"),
    })

    assert [name for name, _ in missing] == ["Absent"]


def test_no_missing_documents_when_all_present(tmp_path):
    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    assert find_missing_documents({"Book": str(pdf)}) == []


# --- load / build are now separate (KNOWN_ISSUES #11) -----------------------

def test_loading_a_missing_index_raises_rather_than_returning_an_empty_one(tmp_path):
    """The old combined function returned a usable-but-empty store here, so a
    missing index looked like a working one that retrieved nothing."""
    with pytest.raises(VectorStoreMissingError, match="scripts/ingest.py"):
        load_vectorstore(str(tmp_path / "nothing"))


def test_building_from_no_documents_is_refused(tmp_path):
    with pytest.raises(ValueError, match="zero documents"):
        build_vectorstore([], str(tmp_path / "store"))


def test_building_over_an_existing_index_is_refused(tmp_path):
    """Chroma appends — verified 6 chunks -> 7 — so this would silently
    duplicate every chunk in the store."""
    existing = tmp_path / "store"
    existing.mkdir()

    with pytest.raises(FileExistsError, match="rebuild"):
        build_vectorstore([Document(page_content="x", metadata={})], str(existing))


def test_the_researcher_no_longer_builds_its_store():
    """Acceptance criterion: `get_vectorstore([])` is gone from the agent."""
    import inspect
    from src.agents import researcher

    code = "\n".join(
        line for line in inspect.getsource(researcher).splitlines()
        if not line.strip().startswith("#")
    )
    assert "get_vectorstore" not in code
    assert "load_vectorstore" in code


# --- the whole pipeline, for real -------------------------------------------

@pytest.mark.slow
def test_ingest_round_trip(tmp_path):
    """load -> split -> embed -> query, on PDFs built here.

    Downloads/loads the embedding model, so it is opt-in: `pytest -m slow`.
    """
    pymupdf = pytest.importorskip("pymupdf")
    from src.data.loader import load_documents

    page_text = (
        "SNEAK ATTACK. Once per turn you can deal an extra 1d6 damage to a "
        "creature you hit if you have advantage on the attack roll. " * 6
    )
    pdf_path = tmp_path / "Players_Handbook_5e.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_textbox(pymupdf.Rect(50, 50, 550, 750), page_text, fontsize=11)
    doc.save(pdf_path)
    doc.close()

    docs = load_documents({"Player's Handbook": str(pdf_path)})
    assert docs and docs[0].metadata["book"] == "Player's Handbook"
    assert "page_number" in docs[0].metadata

    chunks = split_documents(docs)
    store = build_vectorstore(chunks, str(tmp_path / "store"))
    assert store._collection.count() == len(chunks)

    hits = load_vectorstore(str(tmp_path / "store")).as_retriever().invoke(
        "how does sneak attack work"
    )
    assert hits
    assert hits[0].metadata["book"] == "Player's Handbook"
    assert "SNEAK ATTACK" in hits[0].page_content
