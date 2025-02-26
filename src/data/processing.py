from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    """Splits documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(docs)
