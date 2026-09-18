"""Load the source rulebook PDFs, one document per page.

This used to go through `langchain_community.document_loaders.PyMuPDFLoader`.
`langchain-community` is sunset upstream (it warned on every test run), and the
loader was ten lines of PyMuPDF behind a class — so this calls PyMuPDF directly.
Metadata is deliberately small: `book` and `page_number` are what citations
read; `page` is kept for parity with the old loader's 0-based index. The dozen
PyMuPDF document-info fields the old loader copied in (`creationDate`,
`trapped`, `format`...) were only ever prompt noise (KNOWN_ISSUES #15, PR-08).
"""

import logging
from typing import Dict, List

import pymupdf
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_pdf(path: str, book_name: str) -> List[Document]:
    """One `Document` per page of a single PDF, tagged with its book name.

    `page_number` is the 0-based index of the page *in the file*, not the
    number printed on the page — see KNOWN_ISSUES #28.
    """
    documents: List[Document] = []
    with pymupdf.open(path) as pdf:
        total = pdf.page_count
        for index, page in enumerate(pdf):
            documents.append(
                Document(
                    page_content=page.get_text(),
                    metadata={
                        "source": path,
                        "book": book_name,
                        "page": index,
                        "page_number": index,
                        "total_pages": total,
                    },
                )
            )
    return documents


def load_documents(doc_paths: Dict[str, str]) -> List[Document]:
    """Loads every configured PDF. `doc_paths` maps book name -> file path."""
    all_docs: List[Document] = []
    for book_name, path in doc_paths.items():
        logger.info("Loading book: %s", book_name)
        docs = load_pdf(path, book_name)
        logger.info("  %d pages", len(docs))
        all_docs.extend(docs)
    return all_docs
