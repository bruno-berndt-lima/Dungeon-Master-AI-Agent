"""Contract tests for `ResearcherAgent`.

No daemon and no index: `load_vectorstore` is patched before construction, so
these never touch the real store or the embedding model. What is pinned is how
the agent turns retrieved passages into a prompt, when it corrects a query, and
what it promises the player about sources.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

import src.agents.researcher as researcher_module
from src.agents.researcher import RELEVANCE_THRESHOLD, ResearcherAgent

pytestmark = pytest.mark.integration


def doc(text, book="Player's Handbook", page=89):
    metadata = {
        "book": book,
        # The noise the old chain leaked into the prompt, kept here so the
        # formatting test is meaningful.
        "file_path": "Documents/Players_Handbook_5e.pdf",
        "creationDate": "D:20140812013538Z",
        "trapped": "",
        "format": "PDF 1.5",
    }
    if page is not None:
        metadata["page_number"] = page
    return Document(page_content=text, metadata=metadata)


class StubStore:
    """Stands in for the Chroma vector store."""

    def __init__(self, results):
        # results: list of (docs, score) returned in order, one per call
        self.results = list(results)
        self.queries = []

    def similarity_search_with_relevance_scores(self, query, k=4):
        self.queries.append(query)
        docs, score = self.results.pop(0) if self.results else ([], 0.0)
        return [(d, score) for d in docs]

    def as_retriever(self, **kwargs):
        return self


class StubRewriter:
    def __init__(self, result="rewritten question"):
        self.result = result
        self.calls = 0

    def invoke(self, payload, *args, **kwargs):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def make_agent(monkeypatch, results, rewritten="rewritten question"):
    store = StubStore(results)
    monkeypatch.setattr(researcher_module, "load_vectorstore", lambda: store)
    agent = ResearcherAgent()
    agent.rewriter = StubRewriter(rewritten)
    return agent, store


# --- formatting and citations -----------------------------------------------

def test_passages_are_labelled_with_book_and_page(monkeypatch):
    agent, _ = make_agent(monkeypatch, [])
    formatted = agent.format_docs([doc("Sneak Attack text", page=89)])

    assert "[Player's Handbook, p.89]" in formatted
    assert "Sneak Attack text" in formatted


def test_pymupdf_noise_never_reaches_the_prompt(monkeypatch):
    """The old chain interpolated Document objects, so their repr — chunk id,
    file_path, creationDate, trapped, format — became 1,653 tokens of prompt."""
    agent, _ = make_agent(monkeypatch, [])
    formatted = agent.format_docs([doc("rule text")])

    for noise in ["file_path", "creationDate", "trapped", "PDF 1.5", "Document("]:
        assert noise not in formatted


def test_a_chunk_with_no_page_still_gets_a_source(monkeypatch):
    agent, _ = make_agent(monkeypatch, [])
    assert agent.citation_for(doc("t", page=None)) == "Player's Handbook"
    assert agent.citation_for(Document(page_content="t", metadata={})) == "Unknown source"


def test_empty_retrieval_is_stated_not_faked(monkeypatch):
    agent, _ = make_agent(monkeypatch, [])
    assert "no passages" in agent.format_docs([]).lower()


def test_sources_are_appended_from_what_was_retrieved(monkeypatch):
    """Deterministic: the model cited unprompted about half the time."""
    agent, _ = make_agent(monkeypatch, [])
    answer = agent.append_sources("Rogues deal extra damage.", [
        doc("a", page=87), doc("b", page=89), doc("c", book="Monster Manual", page=165),
    ])

    assert "Player's Handbook, p.87" in answer
    assert "Player's Handbook, p.89" in answer
    assert "Monster Manual, p.165" in answer
    assert answer.startswith("Rogues deal extra damage.")


def test_repeated_passages_are_listed_once(monkeypatch):
    agent, _ = make_agent(monkeypatch, [])
    answer = agent.append_sources("x", [doc("a", page=87), doc("b", page=87)])
    assert answer.count("p.87") == 1


def test_no_sources_footer_without_retrieval(monkeypatch):
    agent, _ = make_agent(monkeypatch, [])
    assert agent.append_sources("answer", []) == "answer"


# --- the relevance gate replaces the LLM grader -----------------------------

def test_a_good_match_does_not_pay_for_a_rewrite(monkeypatch):
    """The specced LLM grader cost 31.7 s per query and answered 'yes' every
    time. The retriever's own score is free."""
    agent, store = make_agent(monkeypatch, [([doc("hit")], 0.53)])
    docs, info = agent.retrieve("how does grappling work")

    assert info["relevant"] is True
    assert info["rewritten"] is False
    assert agent.rewriter.calls == 0
    assert store.queries == ["how does grappling work"]


def test_a_poor_match_triggers_one_rewrite_and_retry(monkeypatch):
    agent, store = make_agent(
        monkeypatch,
        [([doc("miss")], 0.09), ([doc("better")], 0.33)],
        rewritten="Can a character use Sneak Attack after Hiding?",
    )
    docs, info = agent.retrieve("can my guy do the thing where he hides and stabs")

    assert info["rewritten"] is True
    assert info["score"] == 0.09
    assert info["retried_score"] == 0.33
    assert info["relevant"] is True
    assert agent.rewriter.calls == 1
    assert store.queries[1] == "Can a character use Sneak Attack after Hiding?"


def test_it_never_rewrites_twice(monkeypatch):
    """One retry, never a loop — this is already the slowest path in the app."""
    agent, store = make_agent(
        monkeypatch, [([doc("miss")], 0.05), ([doc("still bad")], 0.06)]
    )
    agent.retrieve("obscure")

    assert agent.rewriter.calls == 1
    assert len(store.queries) == 2


def test_a_worse_retry_is_discarded(monkeypatch):
    agent, _ = make_agent(
        monkeypatch, [([doc("first", page=10)], 0.20), ([doc("worse", page=99)], 0.01)]
    )
    docs, info = agent.retrieve("something")

    assert [d.metadata["page_number"] for d in docs] == [10]
    assert info["relevant"] is False


def test_the_threshold_sits_between_the_measured_bands():
    """On-topic questions scored 0.363-0.529, off-topic -0.154-0.053."""
    assert 0.053 < RELEVANCE_THRESHOLD < 0.363


def test_a_rewriter_failure_does_not_lose_the_answer(monkeypatch):
    agent, _ = make_agent(monkeypatch, [([doc("miss")], 0.05)])
    agent.rewriter = StubRewriter(RuntimeError("daemon down"))

    docs, info = agent.retrieve("something")
    assert docs
    assert info["rewritten"] is False


def test_no_index_degrades_to_a_bare_answer(monkeypatch):
    monkeypatch.setattr(
        researcher_module, "load_vectorstore",
        lambda: (_ for _ in ()).throw(researcher_module.VectorStoreMissingError("none")),
    )
    agent = ResearcherAgent()

    docs, info = agent.retrieve("anything")
    assert docs == []
    assert info["rag_used"] is False


# --- the node contract ------------------------------------------------------

class StubLLM:
    def __init__(self, result):
        self.result = result

    def invoke(self, messages, *args, **kwargs):
        if isinstance(self.result, Exception):
            raise self.result
        return AIMessage(content=self.result)


def test_the_answer_carries_sources_and_terminates(monkeypatch):
    agent, _ = make_agent(monkeypatch, [([doc("text", page=89)], 0.5)])
    agent.llm = StubLLM("Rogues deal extra damage.")

    command = agent.process_task({
        "current_task": "sneak attack",
        "messages": [HumanMessage(content="sneak attack")],
    })

    assert command.goto == "__end__"
    content = command.update["messages"][0].content
    assert "Player's Handbook, p.89" in content
    assert command.update["messages"][0].name == "researcher"


def test_rag_used_is_recorded(monkeypatch):
    agent, _ = make_agent(monkeypatch, [([doc("text")], 0.5)])
    agent.llm = StubLLM("answer")
    logged = []
    monkeypatch.setattr(agent, "_log_interaction",
                        lambda **kwargs: logged.append(kwargs))

    agent.process_task({"current_task": "q", "messages": [HumanMessage(content="q")]})

    metadata = logged[0]["metadata"]
    assert metadata["rag_used"] is True
    assert metadata["citations"]
    assert "score" in metadata


def test_a_model_failure_still_returns_a_command(monkeypatch):
    agent, _ = make_agent(monkeypatch, [([doc("text")], 0.5)])
    agent.llm = StubLLM(RuntimeError("daemon down"))

    command = agent.process_task({"current_task": "q",
                                  "messages": [HumanMessage(content="q")]})

    assert command.goto == "__end__"
    assert "daemon down" in command.update["messages"][0].content


def test_the_generator_module_is_gone():
    """Dead in every sense: no importers, needed network at construction, and
    `from langchain import hub` no longer imports on LangChain 1.x."""
    with pytest.raises(ImportError):
        import src.pipelines.generator  # noqa: F401
