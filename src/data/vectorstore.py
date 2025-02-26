import os
import logging
from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from ..config import CHROMA_DB_DIRECTORY, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

def get_vectorstore(docs):
    """Loads or creates ChromaDB vector store."""
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    if os.path.exists(CHROMA_DB_DIRECTORY):
        logger.info("Loading existing ChromaDB...")
        return Chroma(persist_directory=CHROMA_DB_DIRECTORY, embedding_function=embedding_model)
    
    logger.info("Creating new ChromaDB...")
    vectorstore = Chroma.from_documents(documents=docs, embedding=embedding_model, persist_directory=CHROMA_DB_DIRECTORY)
    return vectorstore
