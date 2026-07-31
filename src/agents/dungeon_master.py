from typing import Any, Dict, List, Literal

from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.types import Command

from src.agents.base_agent import BaseAgent
from src.models.llm import create_llm
from src.graph.game_state import GameState
from src.prompts.prompts import DUNGEON_MASTER_PROMPT, SCENE_EXTRACTION_PROMPT

# How many prior messages to carry into a narration — two exchanges. The
# checkpointer keeps the whole campaign, but prompt-eval is paid on every turn
# and CPU inference makes it the dominant cost of time-to-first-token: measured
# ~6.6 s at a window of 8, ~3.6 s at 4, ~2.2 s at 2. Continuity does not need the
# transcript, because the durable facts are lifted into `game_state` by the
# extraction pass and fed back as a one-line briefing.
CONTEXT_WINDOW = 4

# Narration is the one place a player watches tokens arrive, and at ~4.4 tok/s a
# runaway answer is a minute of waiting. The prompt asks for two paragraphs; this
# is the hard stop if it does not listen.
MAX_NARRATION_TOKENS = 400

# Marks an LLM call whose tokens are machinery, not story. `main.py` streams by
# node name, and a node may make several calls — this is how it tells them apart.
INTERNAL_TAG = "internal"


class SceneUpdate(TypedDict):
    """Durable world facts extracted from a narration."""
    location: str
    items_gained: List[str]
    effects: List[str]


class DungeonMaster(BaseAgent):
    """Dungeon Master class that manages game interactions."""

    def __init__(self):
        super().__init__("dungeon_master")
        self.llm = create_llm(self.agent_type, temperature=0.8,
                              num_predict=MAX_NARRATION_TOKENS)
        # Separate client: the narration wants warmth, the extraction wants
        # nothing invented. Same model, so no extra memory on the daemon — and
        # measured, `llama3.2:3b` is no faster at this (5.2 s vs 5.0 s) while
        # inventing `effects` out of atmosphere. Nothing to trade.
        self.extractor = create_llm(self.agent_type).with_structured_output(
            SceneUpdate, method="json_schema"
        )
        self.system_prompt = DUNGEON_MASTER_PROMPT

    def get_definition(self) -> str:
        return self.system_prompt

    def _scene_briefing(self, state: GameState) -> str:
        """One line of established world state, or nothing."""
        world = state.get("game_state") or {}
        if not isinstance(world, dict):
            return ""

        parts = []
        if world.get("location"):
            parts.append(f"Current location: {world['location']}.")
        if world.get("inventory"):
            parts.append(f"The player is carrying: {', '.join(world['inventory'])}.")
        if world.get("effects"):
            parts.append(f"In effect: {', '.join(world['effects'])}.")

        return " ".join(parts)

    def _narration_messages(self, state: GameState) -> List[BaseMessage]:
        system = self.system_prompt
        briefing = self._scene_briefing(state)
        if briefing:
            system = f"{system}\n\nEstablished so far: {briefing}"

        history = [
            message
            for message in list(state.get("messages") or [])[-CONTEXT_WINDOW:]
            # A dice result or a rules citation is not part of the story. Feeding
            # them back makes the DM narrate about the mechanics.
            if getattr(message, "name", None) in (None, self.agent_type)
        ]
        if not history:
            history = [HumanMessage(content=self._get_latest_message(state))]

        return [SystemMessage(content=system), *history]

    def _extract_scene(self, narration: str) -> Dict[str, Any]:
        """Pull durable facts out of a narration. Never raises."""
        try:
            update = self.extractor.invoke(
                [
                    SystemMessage(content=SCENE_EXTRACTION_PROMPT),
                    HumanMessage(content=narration),
                ],
                # This call runs inside the same node as the narration, so a
                # consumer streaming by node name cannot tell them apart and
                # would print raw JSON at the player. The tag is that signal.
                config={"tags": [INTERNAL_TAG]},
            )
        except Exception as exc:
            self._log_interaction(
                query=narration,
                response=f"scene extraction failed: {exc}",
                metadata={"error": str(exc), "stage": "extract"},
            )
            return {}

        if not isinstance(update, dict):
            return {}

        return {
            "location": (update.get("location") or "").strip(),
            "items_gained": [i for i in (update.get("items_gained") or []) if i],
            "effects": [e for e in (update.get("effects") or []) if e],
        }

    def _merged_world(self, state: GameState, scene: Dict[str, Any]) -> Dict[str, Any]:
        """Fold a scene update into `game_state` without dropping what was there.

        Returns a new dict — never mutates the graph's own state.
        """
        current = state.get("game_state")
        world = dict(current) if isinstance(current, dict) else {}

        if scene.get("location"):
            world["location"] = scene["location"]

        if scene.get("items_gained"):
            inventory = list(world.get("inventory") or [])
            for item in scene["items_gained"]:
                if item not in inventory:
                    inventory.append(item)
            world["inventory"] = inventory

        if scene.get("effects"):
            effects = list(world.get("effects") or [])
            for effect in scene["effects"]:
                if effect not in effects:
                    effects.append(effect)
            world["effects"] = effects

        return world

    def process_task(self, state: GameState) -> Command[Literal["__end__"]]:
        """Narrates the world's response to what the player did.

        Terminates rather than returning to the supervisor. Every worker does
        since PR-04 — handing back meant the supervisor re-routed on the agent's
        own output (KNOWN_ISSUES #6).
        """
        request = self._get_latest_message(state)
        messages = self._narration_messages(state)

        try:
            # A plain `invoke`. Under `stream_mode="messages"` LangChain routes
            # this through the streaming path anyway, so `main.py` receives
            # tokens as they are produced — first token ~3.5 s, against ~25 s to
            # wait for a finished narration.
            response = self.llm.invoke(messages)
            narration = getattr(response, "content", str(response)).strip()
            if not narration:
                raise ValueError("the model returned an empty narration")
        except Exception as exc:
            error_message = f"The story falters: {exc}"
            self._log_interaction(
                query=request,
                response=error_message,
                metadata={"error": str(exc), "stage": "narrate"},
            )
            return Command(
                goto=END,
                update={
                    "messages": [AIMessage(content=error_message, name=self.agent_type)],
                    "last_response": error_message,
                },
            )

        scene = self._extract_scene(narration)

        self._log_interaction(
            query=request,
            response=narration,
            metadata={"scene": scene, "context_messages": len(messages)},
        )

        update: Dict[str, Any] = {
            "messages": [AIMessage(content=narration, name=self.agent_type)],
            "last_response": narration,
        }
        if any(scene.values()):
            update["game_state"] = self._merged_world(state, scene)

        return Command(goto=END, update=update)
