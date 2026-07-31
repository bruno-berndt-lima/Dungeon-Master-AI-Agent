from typing import List

# `langchain.text_splitter` was removed in LangChain 1.x — the splitters moved to
# their own package. Nothing imported this module before PR-07, so the broken
# import never failed a test run; `scripts/ingest.py` is the first caller.
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# The committed index was built with these values. Changing either changes chunk
# boundaries, so the store has to be rebuilt to stay self-consistent.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def split_documents(
    docs: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """Splits documents into smaller chunks, preserving each one's metadata."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(docs)


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Split raw text. Used by the SRD loader, which re-heads each piece itself."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)
