import logging
from langchain_community.document_loaders import PyMuPDFLoader

logger = logging.getLogger(__name__)

def load_documents(doc_paths):
    """Loads PDFs and returns documents with metadata."""
    all_docs = []
    for book_name, path in doc_paths.items():
        logger.info(f"Loading book: {book_name}")
        loader = PyMuPDFLoader(path)
        docs = loader.load()

        for doc in docs:
            doc.metadata.update({"book": book_name, "page_number": doc.metadata["page"]})
        
        all_docs.extend(docs)
    
    return all_docs
