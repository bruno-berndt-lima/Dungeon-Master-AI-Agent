import re
from typing import Optional, TypedDict, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.types import Command

from src.prompts.prompts import SUPERVISOR_PROMPT
from src.models.llm import create_llm
from src.agents.base_agent import BaseAgent
from src.graph.game_state import GameState

AGENT_TYPES = ["dungeon_master", "researcher", "dice_roller"]

ROUTING_OPTIONS = AGENT_TYPES + ["FINISH"]


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal[*ROUTING_OPTIONS]


# --- deterministic pre-filter ------------------------------------------------
#
# A dice request is one of the few things in this domain with an exact syntax, so
# a regex classifies it *exactly* where a 3B model classifies it *usually*. This
# is a correctness measure (KNOWN_ISSUES #5), not a speed one — warm LLM routing
# already measures ~0.65 s and got 6/6 right in benchmarking. Saving that call is
# a bonus.
#
# The bias is deliberately conservative: the pre-filter only fires when it is
# certain, and falls through to the model whenever it is not. A wrong fast answer
# is worse than a slow right one.

DICE_NOTATION = re.compile(r"\b\d*d\d+\b", re.IGNORECASE)
ROLL_VERB = re.compile(r"\broll(s|ed|ing)?\b", re.IGNORECASE)

# "what does d20 mean?" contains dice notation but is a rules question. Anything
# opening like a question goes to the model, notation or not.
QUESTION_OPENER = re.compile(
    r"^\s*(what|how|why|when|where|which|who|whose|does|do|did|is|are|was|were|"
    r"can|could|should|would|explain|describe|tell\s+me)\b",
    re.IGNORECASE,
)

# A message that is nothing but notation and arithmetic — "2d6+1d8", "d20 + 3".
BARE_NOTATION = re.compile(r"^[\dd\s+\-]+$", re.IGNORECASE)


def prefilter_route(request: str) -> Optional[str]:
    """Route without a model when the request is unambiguously dice.

    Returns an agent type, or None to mean "no confident answer — ask the model".
    """
    text = (request or "").strip()
    if not text:
        return None

    if QUESTION_OPENER.match(text):
        return None

    if not DICE_NOTATION.search(text):
        return None

    # Bare notation, or notation with an explicit roll verb. Both are requests to
    # roll; "my sword does 2d6 slashing" is neither, and goes to the model.
    if BARE_NOTATION.match(text) or ROLL_VERB.search(text):
        return "dice_roller"

    return None


class GameSupervisor(BaseAgent):
    """Supervisor class that manages routing between game agents."""

    def __init__(self):
        super().__init__("supervisor")
        # Constrained decoding against the Router schema. `next` is a Literal, so
        # the model physically cannot emit a destination that is not a real node
        # — which is what retires the old substring-match ladder.
        self.llm = create_llm(self.agent_type).with_structured_output(
            Router, method="json_schema"
        )
        self.system_prompt = SUPERVISOR_PROMPT

    def get_definition(self) -> str:
        return self.system_prompt

    def _routing_request(self, state: GameState) -> str:
        """The text to route on: what the player asked *this turn*.

        Deliberately not the message tail. Routing on the tail is KNOWN_ISSUES #6:
        after `dice_roller` appends its result, the newest message is the roll, so
        the supervisor routed on the *answer* instead of the question and sent it
        to `researcher` to explain a roll nobody asked about.
        """
        task = (state.get("current_task") or "").strip()
        if task:
            return task

        for message in reversed(list(state.get("messages") or [])):
            if isinstance(message, HumanMessage):
                return str(message.content).strip()

        return ""

    def process_task(self, state: GameState) -> Command[Literal[*AGENT_TYPES, "__end__"]]:
        request = self._routing_request(state)

        shortcut = prefilter_route(request)
        if shortcut is not None:
            self._log_interaction(
                query=request,
                response=shortcut,
                metadata={"routed_to": shortcut, "router": "prefilter"},
            )
            return Command(goto=shortcut, update={"active_agent": shortcut})

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=request),
        ]

        try:
            decision = self.llm.invoke(messages)
            goto = decision["next"] if isinstance(decision, dict) else None
            if goto not in ROUTING_OPTIONS:
                raise ValueError(f"router returned {decision!r}")
        except Exception as exc:
            # No silent fallback. The old code sent every failure to `researcher`,
            # so a dead daemon or an unparseable reply became a confident-looking
            # RAG answer to a question the player never asked. Say so instead.
            self._log_interaction(
                query=request,
                response=f"Error: {exc}",
                metadata={"error": str(exc), "routed_to": "__end__", "router": "llm"},
            )
            return Command(
                goto=END,
                update={
                    "messages": [
                        AIMessage(
                            content=(
                                "I could not work out which part of the table should "
                                "handle that. Try rephrasing it as a rules question, "
                                "a narrative action, or a dice roll."
                            ),
                            name=self.agent_type,
                        )
                    ],
                    "active_agent": self.agent_type,
                },
            )

        self._log_interaction(
            query=request,
            response=goto,
            metadata={"routed_to": goto, "router": "llm"},
        )

        if goto == "FINISH":
            # Ending the turn with no message at all reads as the app having
            # hung. A fixed string, not a generation — there is nothing to say
            # that is worth 40 s.
            return Command(
                goto=END,
                update={
                    "messages": [
                        AIMessage(
                            content="Ready when you are, adventurer.",
                            name=self.agent_type,
                        )
                    ],
                    "active_agent": "FINISH",
                },
            )

        return Command(goto=goto, update={"active_agent": goto})
