"""Campaign memory: a rolling journal instead of a growing transcript.

The checkpointer keeps every message of a campaign, but the Dungeon Master
must not read them all: prompt-eval dominates time-to-first-token on CPU
(KNOWN_ISSUES #26), and a level-1 party talks a lot. So the DM sees the scene
sheet, a short journal, and the last few messages — and this node keeps the
journal.

It runs after every narration. Once the transcript passes `MAX_MESSAGES`, the
oldest messages beyond `KEEP_RECENT` are rendered as a transcript, one
`internal`-tagged model call rewrites the journal to include them, and the
folded messages are removed from state with `RemoveMessage`. Below the
threshold it does nothing and costs nothing. If the model call fails, nothing
is removed — the transcript stays until the next narration, when it is tried
again. Losing history is worse than a longer prompt.

Why a model call at all: what matters later is not which messages were said
but what they established — a name, a promise, a wound, a door left barred.
Extracting that is a language task. It is also the only model call in this
project whose output nobody reads at the table, so it does not stream.
"""

from typing import Any, List, Literal

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END
from langgraph.types import Command

from src.agents.base_agent import BaseAgent
from src.graph.game_state import GameState
from src.models.llm import create_llm
from src.prompts.prompts import SUMMARY_PROMPT

# Fold once the transcript is this long...
MAX_MESSAGES = 24
# ...keeping this many of the newest messages verbatim. The DM's own context
# window is smaller than this, so what it reads is never half-folded.
KEEP_RECENT = 10
# The journal's own hard stop, in tokens; a journal that grows without bound
# would defeat the purpose.
MAX_SUMMARY_TOKENS = 350

INTERNAL_TAG = "internal"


def transcript(messages: List[BaseMessage]) -> str:
    """Old messages as a readable script. Tool traffic becomes table notes."""
    lines: List[str] = []
    for message in messages:
        content = str(getattr(message, "content", "") or "").strip()
        if isinstance(message, HumanMessage):
            who = getattr(message, "name", None) or "Player"
            lines.append(f"{who}: {content}")
        elif isinstance(message, ToolMessage):
            if content:
                lines.append(f"[table] {content}")
        elif isinstance(message, AIMessage):
            if getattr(message, "tool_calls", None) and not content:
                continue
            who = getattr(message, "name", None) or "DM"
            if content:
                lines.append(f"{'DM' if who == 'dungeon_master' else who}: {content}")
    return "\n".join(lines)


class Memory(BaseAgent):
    def __init__(self, llm: Any = None):
        super().__init__("memory")
        self.llm = llm if llm is not None else create_llm(
            self.agent_type, temperature=0.2, num_predict=MAX_SUMMARY_TOKENS
        )

    def get_definition(self) -> str:
        return SUMMARY_PROMPT

    def fold(self, summary: str, old: List[BaseMessage]) -> str:
        """The journal, rewritten to include what `old` established."""
        script = transcript(old)
        if not script.strip():
            return summary
        messages = [
            SystemMessage(content=SUMMARY_PROMPT),
            HumanMessage(
                content=f"Journal so far:\n{summary.strip() or '(nothing yet)'}\n\nWhat happened next:\n{script}\n\nWrite the updated journal."
            ),
        ]
        response = self.llm.invoke(messages, config={"tags": [INTERNAL_TAG]})
        text = str(getattr(response, "content", "") or "").strip()
        return text or summary

    def process_task(self, state: GameState) -> Command[Literal["__end__"]]:
        history = list(state.get("messages") or [])
        if len(history) <= MAX_MESSAGES:
            return Command(goto=END)

        cutoff = len(history) - KEEP_RECENT
        old, summary = history[:cutoff], (state.get("summary") or "")
        try:
            updated = self.fold(summary, old)
        except Exception as exc:
            self._log_interaction(
                query=f"fold {len(old)} messages",
                response=f"fold failed: {exc}",
                metadata={"error": str(exc), "stage": "fold", "folded": 0},
            )
            return Command(goto=END)

        self._log_interaction(
            query=transcript(old)[:2000],
            response=updated,
            metadata={"stage": "fold", "folded": len(old), "kept": KEEP_RECENT, "summary_chars": len(updated)},
        )
        removals = [RemoveMessage(id=m.id) for m in old if getattr(m, "id", None)]
        return Command(goto=END, update={"summary": updated, "messages": removals})
