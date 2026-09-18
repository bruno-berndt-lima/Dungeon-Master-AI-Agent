"""The Dungeon Master: a narrator that uses tools.

Two graph nodes live on this class. `process_task` ("dungeon_master") calls
the model with the scene sheet and the tools bound; if the model asks for
tools the turn goes to `run_tools` ("tools"), which runs them through the
engine and comes back. The loop is capped at `MAX_TOOL_STEPS` per turn, and a
turn always ends with narration.

What the model is *not* allowed to do is decide a number. It picks a tool; the
engine answers; the narration is built from the answer. Two situations force
narrate-only mode (no tools bound): a roll is pending — the DM asked a player
for a check and must wait — and the tool budget is spent.

Streaming, and why the planner's prose is buffered: measured on qwen2.5:7b,
a reply that ends in a tool call sometimes *starts* with prose that assumes
the tool's outcome ("You hit the goblin with a solid blow!" before `attack`
ran). Text whose validity depends on how the reply ends cannot be streamed
honestly, so `main.py` buffers the planner's tokens and shows them only if
the reply carried no tool call; the content of a tool-call reply is dropped
here too, so it never reaches state or the next prompt. Calls that are known
in advance to be narration (narrate-only mode) are tagged `narration` and
stream live. KNOWN_ISSUES #29 holds the trade-off; PR-20 measures it.
"""

from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END
from langgraph.types import Command

from src.agents.base_agent import BaseAgent
from src.graph.game_state import GameState
from src.models.llm import create_llm
from src.prompts.prompts import DUNGEON_MASTER_PROMPT, NARRATE_ONLY_NOTE
from src.tools import run_tool, tool_schemas
from src.tools.tools import get_scene
from src.utils.dice import RandomSource

# Prior narrative messages carried into a call. Prompt-eval is the dominant cost
# of time-to-first-token on CPU (KNOWN_ISSUES #26); the scene sheet carries the
# state, so the transcript can stay short.
CONTEXT_WINDOW = 6

# Narration is the one place a player watches tokens arrive; this is the hard
# stop if the prompt's "two paragraphs" is ignored.
MAX_NARRATION_TOKENS = 400

# Tool calls per turn. Enough for start_encounter → attack → end_turn → narrate;
# a model that keeps asking for tools past this narrates what it has.
MAX_TOOL_STEPS = 4

SILENT_FALLBACK = "The Dungeon Master pauses, gathering the thread of the story. What do you do?"

# Tag for a model call that is narration for certain, so the front end may
# stream it token by token. Planner calls carry no tag and are buffered.
NARRATION_TAG = "narration"


def _is_narrative(message: BaseMessage) -> bool:
    """A message the table saw: a player's words, or a reply without tool calls."""
    if isinstance(message, HumanMessage):
        return True
    if isinstance(message, AIMessage):
        return not getattr(message, "tool_calls", None)
    return False


class DungeonMaster(BaseAgent):
    def __init__(self, llm: Any = None, rng: Optional[RandomSource] = None):
        super().__init__("dungeon_master")
        base = llm if llm is not None else create_llm(
            self.agent_type, temperature=0.8, num_predict=MAX_NARRATION_TOKENS
        )
        self.narrator = base
        self.planner = base.bind_tools(tool_schemas())
        self.rng = rng

    def get_definition(self) -> str:
        return DUNGEON_MASTER_PROMPT

    # --- prompt assembly --------------------------------------------------------

    def _system_prompt(self, state: GameState, narrate_only_reason: Optional[str]) -> str:
        parts = [DUNGEON_MASTER_PROMPT, "## The table right now", get_scene(state).text]
        summary = (state.get("summary") or "").strip()
        if summary:
            parts += ["## The story so far", summary]
        text = "\n\n".join(parts)
        if narrate_only_reason:
            text += NARRATE_ONLY_NOTE.format(reason=narrate_only_reason)
        return text

    def _messages(self, state: GameState, narrate_only_reason: Optional[str]) -> List[BaseMessage]:
        history = list(state.get("messages") or [])
        last_human = max((i for i, m in enumerate(history) if isinstance(m, HumanMessage)), default=None)
        if last_human is None:
            prior, current = history, []
        else:
            prior, current = history[:last_human], history[last_human:]
        recent = [m for m in prior if _is_narrative(m)][-CONTEXT_WINDOW:]
        # `current` is this turn: the player's message and the tool exchange so
        # far. It is kept whole so the model sees what its tools answered.
        # A result `intake` resolved this turn (a declared attack, a roll) is
        # presented as a report from the table to narrate — measured, the
        # model treats a trailing assistant message as already said and replies
        # with a bare "What will you do?".
        current = [
            HumanMessage(content=f"[Results from the table — narrate these]\n{m.content}", name="table")
            if isinstance(m, AIMessage) and getattr(m, "name", None) == "intake"
            else m
            for m in current
        ]
        return [SystemMessage(content=self._system_prompt(state, narrate_only_reason)), *recent, *current]

    def _narrate_only_reason(self, state: GameState) -> Optional[str]:
        if state.get("pending"):
            return "a player still owes you a roll"
        if int(state.get("tool_steps") or 0) >= MAX_TOOL_STEPS:
            return "you have used every tool call this turn allows"
        return None

    # --- the two nodes -----------------------------------------------------------

    def process_task(self, state: GameState) -> Command[Literal["tools", "__end__"]]:
        """Ask the model what happens; go to `tools` if it needs them, else narrate."""
        reason = self._narrate_only_reason(state)
        messages = self._messages(state, reason)
        model = self.narrator if reason else self.planner
        request = str(messages[-1].content) if messages[-1:] else ""
        steps = int(state.get("tool_steps") or 0)

        try:
            response = model.invoke(messages, config={"tags": [NARRATION_TAG]} if reason else None)
        except Exception as exc:
            error = f"The story falters: {exc}"
            self._log_interaction(query=request, response=error, metadata={"error": str(exc), "stage": "narrate"})
            return Command(goto=END, update={"messages": [AIMessage(content=error, name=self.agent_type)], "last_response": error, "tool_steps": 0})

        tool_calls = list(getattr(response, "tool_calls", None) or [])
        if tool_calls and not reason:
            self._log_interaction(
                query=request,
                response=f"tool calls: {[(c['name'], c['args']) for c in tool_calls]}",
                metadata={
                    "stage": "plan", "tool_steps": steps + 1, "tools": [c["name"] for c in tool_calls],
                    "discarded_prose": str(getattr(response, "content", "") or "")[:200],
                },
            )
            # Prose that came with a tool call assumed the tool's outcome. It
            # is not narration and must not reach the next prompt.
            request_only = AIMessage(content="", tool_calls=tool_calls, id=getattr(response, "id", None))
            return Command(goto="tools", update={"messages": [request_only], "tool_steps": steps + 1})

        narration = str(getattr(response, "content", "") or "").strip() or SILENT_FALLBACK
        self._log_interaction(
            query=request,
            response=narration,
            metadata={"stage": "narrate", "tool_steps": steps, "narrate_only": reason, "context_messages": len(messages)},
        )
        return Command(
            goto=END,
            update={"messages": [AIMessage(content=narration, name=self.agent_type)], "last_response": narration, "tool_steps": 0},
        )

    def run_tools(self, state: GameState) -> Command[Literal["dungeon_master"]]:
        """Run every tool the model asked for, in order, each seeing the last's effect."""
        history = list(state.get("messages") or [])
        request = history[-1] if history else None
        calls = list(getattr(request, "tool_calls", None) or [])

        working: Dict[str, Any] = dict(state)
        updates: Dict[str, Any] = {}
        replies: List[ToolMessage] = []
        for call in calls:
            result = run_tool(working, call.get("name", ""), call.get("args") or {}, rng=self.rng)
            working.update(result.update)
            updates.update(result.update)
            replies.append(
                ToolMessage(
                    content=result.text,
                    tool_call_id=call.get("id") or call.get("name", ""),
                    name=call.get("name", ""),
                    additional_kwargs={"args": call.get("args") or {}, "ok": result.ok},
                )
            )
            self._log_interaction(
                query=f"{call.get('name')}({call.get('args')})",
                response=result.text,
                metadata={"stage": "tool", "tool": call.get("name"), "ok": result.ok, "events": [e.kind for e in result.events]},
            )

        if not replies:
            replies.append(ToolMessage(content="No tool was named. Narrate.", tool_call_id="none", name="none"))
        return Command(goto="dungeon_master", update={**updates, "messages": replies})
