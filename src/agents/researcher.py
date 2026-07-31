import warnings
from typing import Any, Dict, List, Literal, Tuple

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END
from langgraph.types import Command

from src.agents.base_agent import BaseAgent
from src.data.vectorstore import VectorStoreMissingError, load_vectorstore
from src.graph.game_state import GameState
from src.models.llm import create_llm
from src.pipelines.rewriter import create_question_rewriter
from src.prompts.prompts import RESEARCHER_PROMPT

# Rules answers are read, not skimmed, and every token costs ~0.25 s here. One
# unbounded answer measured 461 tokens and 181 s.
MAX_ANSWER_TOKENS = 400

# Chunks per query. Each is ~1000 characters, so this is the dominant term in
# prompt-eval time — which is what the player experiences as silence.
RETRIEVAL_K = 4

# Below this, the best retrieved passage is treated as a miss and the question is
# restated once. Measured over this index: on-topic questions score 0.363-0.529,
# off-topic ones -0.154-0.053. The gap is wide, so the exact value is not
# delicate — but it is specific to this index and embedding model, and would
# need re-measuring after a corpus change.
RELEVANCE_THRESHOLD = 0.25

# Marks the rewriter call, which runs inside this node but is not for the
# player. `main.py` streams by node name and would otherwise print it.
INTERNAL_TAG = "internal"


class ResearcherAgent(BaseAgent):
    """Agent that provides information about D&D rules and lore."""

    def __init__(self):
        super().__init__("researcher")
        self.llm = create_llm(self.agent_type, num_predict=MAX_ANSWER_TOKENS)
        self.system_prompt = RESEARCHER_PROMPT

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt + "\n\nRetrieved passages:\n{context}"),
            ("user", "{question}"),
        ])

        # Written in early 2025 and never wired in. The retrieval *grader* is
        # not used — see `_retrieve_scored` for why the retriever's own score
        # replaced it — but the rewriter earns its call when retrieval misses.
        self.rewriter = create_question_rewriter(self.llm)

        try:
            # Read-only. This used to be `get_vectorstore([])` — passing an
            # empty document list to a build-or-load function and relying on it
            # to ignore the argument (KNOWN_ISSUES #11). If no index exists this
            # now raises instead of handing back an empty store that retrieves
            # nothing while looking healthy.
            self.vectorstore = load_vectorstore()
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": RETRIEVAL_K}
            )
        except VectorStoreMissingError as exc:
            print(f"Warning: {exc} Answering without retrieval.")
            self.vectorstore = None
            self.retriever = None
        except Exception as e:
            print(f"Warning: Could not initialize vectorstore: {e}")
            self.vectorstore = None
            self.retriever = None

    def get_definition(self) -> str:
        return "I am a researcher assistant that provides information about D&D rules, lore, monsters, spells, and game mechanics."

    @staticmethod
    def citation_for(doc: Document) -> str:
        """A human-checkable source label for one chunk.

        Two corpora produce two shapes. The SRD index carries the entry's own
        name, which is the better citation — a reader can look up
        `Monsters: Goblin`, whereas a PDF page index is offset from the printed
        page number and cannot be checked (KNOWN_ISSUES #28).
        """
        metadata = doc.metadata
        name = metadata.get("name")
        if name:
            source = metadata.get("source") or "Unknown source"
            category = metadata.get("category")
            return f"{source}, {category}: {name}" if category else f"{source}: {name}"

        book = metadata.get("book") or "Unknown source"
        page = metadata.get("page_number")
        return f"{book}, p.{page}" if page is not None else book

    def format_docs(self, docs: List[Document]) -> str:
        """Render retrieved chunks as labelled passages.

        Defined since the first commit and never wired into the chain, which
        instead interpolated the `Document` objects themselves. That put their
        `repr` in the prompt — chunk id, `file_path`, `creationDate`, `trapped`,
        `format`, and a dozen other PyMuPDF fields — 1,653 tokens of prompt for
        four chunks, most of it noise. `book` and `page_number` were in there,
        buried, which is why answers cited plausible page numbers rather than
        real ones: for "sneak attack" the model said p.94 while the retrieved
        chunks were p.87, p.89 and p.90.
        """
        if not docs:
            return "(no passages retrieved)"
        return "\n\n".join(
            f"[{self.citation_for(doc)}]\n{doc.page_content.strip()}" for doc in docs
        )

    def append_sources(self, answer: str, docs: List[Document]) -> str:
        """Append the passages the answer was built from.

        Deterministic, because we already know them. Asking the model to cite
        works about half the time — it cited `Monster Manual, p.165` unprompted
        for one question and cited nothing for the next. There is no reason to
        depend on that for a fact this code holds: the prompt still asks for
        inline citation, and this guarantees the player can check the answer
        either way.
        """
        if not docs:
            return answer

        seen = []
        for doc in docs:
            citation = self.citation_for(doc)
            if citation not in seen:
                seen.append(citation)

        listed = "\n".join(f"- {citation}" for citation in seen)
        return f"{answer.rstrip()}\n\n---\n**Passages consulted:**\n{listed}"

    def _retrieve_scored(self, question: str) -> Tuple[List[Document], float]:
        """Retrieve, and report how well the best passage matched.

        The score *is* the relevance grade. PR-08 originally wired
        `create_retrieval_grader` here, as specced — one structured-output call
        over the whole retrieved set. Measured on this machine it cost **31.7 s**
        and returned "yes" for every query tried, because it re-evaluates the
        same ~1,000-token context the answer call is about to evaluate again.

        The retriever already knows. Measured over this index, on-topic
        questions score 0.363–0.529 and off-topic ones -0.154–0.053, so a
        threshold separates them with room to spare, for free.
        """
        with warnings.catch_warnings():
            # Chroma warns when a cosine distance maps outside [0, 1]. Expected
            # here, and the ordering is what matters.
            warnings.simplefilter("ignore", UserWarning)
            scored = self.vectorstore.similarity_search_with_relevance_scores(
                question, k=RETRIEVAL_K
            )
        if not scored:
            return [], 0.0
        return [doc for doc, _ in scored], max(score for _, score in scored)

    def _rewrite(self, question: str) -> str:
        """Restate a question in rulebook language. Returns the original on failure."""
        try:
            rewritten = self.rewriter.invoke(
                {"question": question}, config={"tags": [INTERNAL_TAG]}
            )
            rewritten = str(rewritten).strip()
            return rewritten or question
        except Exception as exc:
            self._log_interaction(
                query=question,
                response=f"rewrite failed: {exc}",
                metadata={"error": str(exc), "stage": "rewrite"},
            )
            return question

    def retrieve(self, question: str) -> Tuple[List[Document], Dict[str, Any]]:
        """Retrieve passages, correcting the query once if the first try misses.

        Returns the passages and a metadata dict describing what happened, which
        goes straight into the JSONL log — the corrective path is invisible
        otherwise.
        """
        info: Dict[str, Any] = {"rag_used": False, "rewritten": False}
        if self.vectorstore is None:
            return [], info

        docs, score = self._retrieve_scored(question)
        info.update(rag_used=True, retrieved=len(docs), score=round(score, 3))

        if score >= RELEVANCE_THRESHOLD:
            info["relevant"] = True
            info["citations"] = [self.citation_for(d) for d in docs]
            return docs, info

        # One retry, never a loop. Player phrasing and rulebook phrasing sit far
        # apart in embedding space, so a restatement is worth one model call —
        # but only when the first attempt actually missed.
        info["relevant"] = False
        rewritten = self._rewrite(question)
        if rewritten == question:
            info["citations"] = [self.citation_for(d) for d in docs]
            return docs, info

        retried, retried_score = self._retrieve_scored(rewritten)
        info.update(
            rewritten=True,
            rewritten_query=rewritten,
            retried_score=round(retried_score, 3),
        )

        # Keep whichever attempt actually matched better.
        if retried and retried_score > score:
            docs = retried
            info["relevant"] = retried_score >= RELEVANCE_THRESHOLD

        info["citations"] = [self.citation_for(d) for d in docs]
        return docs, info

    def process_task(self, state: GameState) -> Command[Literal["__end__"]]:
        """Retrieves and provides D&D-related information.

        The annotation says ``__end__`` because that is what this method
        returns. It previously claimed ``supervisor``; LangGraph derives a
        node's legal destinations from this annotation, so the mismatch was a
        latent bug rather than a documentation slip.
        """
        latest_message = self._get_latest_message(state)

        try:
            docs, info = self.retrieve(latest_message)

            if docs:
                messages = self.prompt_template.invoke({
                    "context": self.format_docs(docs),
                    "question": latest_message,
                })
            else:
                # No index, or nothing retrieved. Answer from the model alone
                # and say so — `metadata.rag_used` records which path ran.
                messages = [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=latest_message),
                ]

            # A plain `invoke`: under `stream_mode="messages"` LangChain routes
            # it through the streaming path, so the answer reaches the player
            # token by token instead of arriving whole after a minute.
            response = self.llm.invoke(messages)
            response_content = StrOutputParser().invoke(response)
            response_content = self.append_sources(response_content, docs)

            self._log_interaction(
                query=latest_message,
                response=response_content,
                metadata=info,
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

            self._log_interaction(
                query=latest_message,
                response=error_message,
                metadata={"error": str(e)},
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
