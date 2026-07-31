from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def create_question_rewriter(llm):
    """Creates and returns a question rewriter chain.

    Written in early 2025 and never wired in until PR-08. `ResearcherAgent`
    calls it only when the first retrieval scores below its relevance
    threshold — player phrasing and rulebook phrasing sit far apart in
    embedding space. Measured: "can my guy do the thing where he hides and
    stabs" scored 0.09, and 0.334 after rewriting.

    The output goes straight into an embedding lookup, so it has to be a bare
    question. The original prompt asked the model to "formulate an improved
    question" and got commentary with it — "This version clarifies that you're
    referring to..." — which was then embedded along with the question.
    """
    system = """You rewrite a player's casual question into the vocabulary the
D&D 5e rulebooks use, so it can be matched against them.

Use the game's own terms: "hides and stabs" is Sneak Attack and Hiding,
"knocked out" is Unconscious and Death Saving Throws, "tougher" is Armor Class,
Hit Points and Constitution.

Reply with the rewritten question and nothing else — no preamble, no
explanation, no quotation marks. One sentence."""

    re_write_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{question}"),
    ])

    return re_write_prompt | llm | StrOutputParser()
