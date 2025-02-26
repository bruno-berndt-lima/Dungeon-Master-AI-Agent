from langchain_ollama import ChatOllama

def create_llm(model_name="Llama3.2", temperature=0):
    """Create and return LLM instances."""
    return ChatOllama(model=model_name, temperature=temperature)

def create_json_llm(model_name="Llama3.2", temperature=0):
    """Create and return JSON-mode LLM instance."""
    return ChatOllama(model=model_name, temperature=temperature, format="json")