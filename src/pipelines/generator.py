from langchain import hub
from langchain_core.output_parsers import StrOutputParser

def create_rag_chain(llm):
    """Creates and returns a RAG chain."""
    prompt = hub.pull("rlm/rag-prompt")
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    return prompt | llm | StrOutputParser()