import logging
import shutil
from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from ..config import CHROMA_DB_DIRECTORY, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


class VectorStoreMissingError(FileNotFoundError):
    """No index on disk, and the caller asked to read rather than build one."""


def create_embeddings(model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddings:
    """The embedding model. Changing it invalidates the entire index.

    `all-MiniLM-L6-v2` produces 384-dim vectors, and the committed store is built
    from them — a store built with one model cannot be queried with another.
    """
    return HuggingFaceEmbeddings(model_name=model_name)


def load_vectorstore(persist_directory: str = CHROMA_DB_DIRECTORY) -> Chroma:
    """Open an existing index. Never builds one.

    Raises:
        VectorStoreMissingError: if nothing is on disk. The old combined
            function returned an *empty but usable* store here, so a missing
            index looked like a working one that just never retrieved anything.
    """
    if not Path(persist_directory).exists():
        raise VectorStoreMissingError(
            f"No vector index at {persist_directory!r}. Build one with "
            f"`python scripts/ingest.py`."
        )

    logger.info("Loading existing ChromaDB from %s", persist_directory)
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=create_embeddings(),
    )


def build_vectorstore(
    docs: List[Document],
    persist_directory: str = CHROMA_DB_DIRECTORY,
    rebuild: bool = False,
) -> Chroma:
    """Index `docs` into a store at `persist_directory`.

    Args:
        rebuild: delete any existing index first. Without it, building over one
            that already exists is refused — Chroma appends, which would
            silently duplicate every chunk.
    """
    if not docs:
        raise ValueError("refusing to build an index from zero documents")

    path = Path(persist_directory)
    if path.exists():
        if not rebuild:
            raise FileExistsError(
                f"An index already exists at {persist_directory!r}. Pass "
                f"rebuild=True (or `--rebuild`) to replace it; building over it "
                f"would append a duplicate of every chunk."
            )
        logger.info("Removing existing ChromaDB at %s", persist_directory)
        shutil.rmtree(path)

    logger.info("Creating new ChromaDB with %d chunks", len(docs))
    return Chroma.from_documents(
        documents=docs,
        embedding=create_embeddings(),
        persist_directory=persist_directory,
    )


def get_vectorstore(docs: Optional[List[Document]] = None) -> Chroma:
    """Deprecated — use `load_vectorstore()` or `build_vectorstore(docs)`.

    Kept only so an outside caller does not break on the rename. Its old
    behaviour, silently discarding `docs` whenever the directory existed, is
    exactly why `ResearcherAgent` had to call it with `[]` (KNOWN_ISSUES #11).
    """
    if Path(CHROMA_DB_DIRECTORY).exists():
        return load_vectorstore()
    return build_vectorstore(docs or [])
