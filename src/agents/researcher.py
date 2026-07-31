from typing import Literal, Dict, Any, List
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import END
from langgraph.types import Command
from src.data.vectorstore import get_vectorstore
from src.prompts.prompts import RESEARCHER_PROMPT
from src.models.llm import create_llm
from src.agents.base_agent import BaseAgent
from src.graph.game_state import GameState

class ResearcherAgent(BaseAgent):
    """Agent that provides information about D&D rules and lore."""
    
    def __init__(self):
        super().__init__("researcher")
        self.llm = create_llm()
        self.system_prompt = RESEARCHER_PROMPT
        
        # Initialize vectorstore retriever
        try:
            self.vectorstore = get_vectorstore([])  # Empty list to load existing store
            self.retriever = self.vectorstore.as_retriever()
            
            # Create RAG chain
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt + "\n\nRelevant context:\n{context}"),
                ("user", "{question}")
            ])
            
            self.rag_chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | prompt_template
                | self.llm
                | StrOutputParser()
            )
        except Exception as e:
            print(f"Warning: Could not initialize vectorstore: {e}")
            self.retriever = None
            self.rag_chain = None
    
    def get_definition(self) -> str:
        return "I am a researcher assistant that provides information about D&D rules, lore, monsters, spells, and game mechanics."
    
    def format_docs(self, docs: List[Document]) -> str:
        """Format documents into a single context string."""
        if not docs:
            return ""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def process_task(self, state: GameState) -> Command[Literal["__end__"]]:
        """Retrieves and provides D&D-related information.

        The annotation says ``__end__`` because that is what this method
        returns. It previously claimed ``supervisor``; LangGraph derives a
        node's legal destinations from this annotation, so the mismatch was a
        latent bug rather than a documentation slip.
        """
        # Extract the latest message from the state
        latest_message = self._get_latest_message(state)
        
        # Get response using RAG chain or fallback to direct LLM if retriever not available
        try:
            if self.rag_chain:
                response_content = self.rag_chain.invoke(latest_message)
            else:
                # Fallback to direct LLM if retriever is not available
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": latest_message}
                ]
                response = self.llm.invoke(messages)
                response_content = response.content if hasattr(response, "content") else str(response)
                
            # Log the interaction
            self._log_interaction(
                query=latest_message,
                response=response_content,
                metadata={"rag_used": self.rag_chain is not None}
            )
            
            # Return only the message this node produced — the add_messages
            # reducer appends it. Returning the whole history would duplicate it.
            return Command(
                goto="__end__",
                update={
                    "messages": [
                        AIMessage(content=response_content, name=self.agent_type)
                    ],
                    "last_response": response_content,
                },
            )
            
        except Exception as e:
            error_message = f"Error researching D&D information: {str(e)}"
            
            # Log the error
            self._log_interaction(
                query=latest_message,
                response=error_message,
                metadata={"error": str(e)}
            )
            
            return Command(
                goto="__end__",
                update={
                    "messages": [
                        AIMessage(content=error_message, name=self.agent_type)
                    ],
                    "last_response": error_message,
                },
            )
