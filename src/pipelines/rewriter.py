from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def create_question_rewriter(llm):
    """Creates and returns a question rewriter chain."""
    
    system = """You are a question re-writer that converts an input question into a more effective version, specifically for Dungeons & Dragons scenarios.
         Analyze the input question and enhance it to better reflect the context, rules, and mechanics of D&D gameplay, ensuring it captures the underlying intent and thematic elements while adhering to the game's rules."""
    
    re_write_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
    ])
    
    return re_write_prompt | llm | StrOutputParser()