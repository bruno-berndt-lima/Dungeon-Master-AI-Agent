"""Campaigns are checkpointer threads. This module names them and lists them.

The checkpointer has held every turn of every session since PR-03; what was
missing was a way to get back to one. `main.py` used to mint a fresh UUID per
run, so a campaign could be resumed in principle and never was in practice.

Everything here reads the checkpointer through its public `list` / snapshot
interface rather than the SQLite tables, so it will work unchanged over the
async saver the Discord bot needs (ROADMAP PR-22), where a thread is a channel.
"""

import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver

from src.graph.game_state import create_default_game_state

# Eight hex characters: short enough to type after `--thread`, and with ten
# campaigns in the database the collision odds are one in hundreds of millions.
THREAD_ID_LENGTH = 8


@dataclass
class CampaignSummary:
    """What a player needs to recognise a campaign in a list."""

    thread_id: str
    last_active: str  # ISO 8601, from the newest checkpoint
    turns: int  # player messages so far
    party: List[str]  # character names
    last_prompt: str  # the player's most recent input, or ""


def new_thread_id() -> str:
    return uuid.uuid4().hex[:THREAD_ID_LENGTH]


def _last_of(messages: Iterable, kind, name: Optional[str] = None) -> str:
    for message in reversed(list(messages or [])):
        if isinstance(message, kind) and (name is None or getattr(message, "name", None) == name):
            return str(message.content)
    return ""


def summarize(thread_id: str, values: Dict[str, Any], last_active: str = "") -> CampaignSummary:
    """A summary from a thread's latest state values."""
    messages = list(values.get("messages") or [])
    party = values.get("party") or {}
    return CampaignSummary(
        thread_id=thread_id,
        last_active=last_active,
        turns=sum(isinstance(m, HumanMessage) for m in messages),
        party=[str(c.get("name", key)) for key, c in party.items()] if isinstance(party, dict) else [],
        last_prompt=_last_of(messages, HumanMessage),
    )


def list_campaigns(checkpointer: BaseCheckpointSaver) -> List[CampaignSummary]:
    """Every thread in the checkpointer, newest activity first.

    `list(None)` yields every checkpoint of every thread, newest first, so the
    first tuple seen for a thread is its current state.
    """
    newest: Dict[str, CampaignSummary] = {}
    for checkpoint_tuple in checkpointer.list(None):
        thread_id = checkpoint_tuple.config["configurable"]["thread_id"]
        if thread_id in newest:
            continue
        checkpoint = checkpoint_tuple.checkpoint
        newest[thread_id] = summarize(
            thread_id,
            checkpoint.get("channel_values") or {},
            last_active=str(checkpoint.get("ts") or ""),
        )
    return sorted(newest.values(), key=lambda c: c.last_active, reverse=True)


def is_new_campaign(values: Dict[str, Any]) -> bool:
    """A thread with no checkpoint yet has no state values at all."""
    return not values


def seed_turn(turn: Dict[str, Any], existing_values: Dict[str, Any]) -> Dict[str, Any]:
    """The payload for one turn.

    A brand-new thread needs the full default state under the new message. A
    resumed one must get *only* the new message: merging the defaults in again
    would overwrite the stored `game_state` with `{}` — every other field
    replaces on write, only `messages` has a reducer — and the campaign would
    forget where the party is.
    """
    if is_new_campaign(existing_values):
        return {**create_default_game_state(), **turn}
    return dict(turn)


def recap(values: Dict[str, Any]) -> str:
    """A few lines that put a returning player back in the scene. No model call."""
    if is_new_campaign(values):
        return ""
    summary = summarize("", values)
    lines = []
    if summary.turns:
        lines.append(f"{summary.turns} turn{'s' if summary.turns != 1 else ''} so far.")
    if summary.party:
        lines.append(f"Party: {', '.join(summary.party)}.")
    journal = str(values.get("summary") or "").strip()
    if journal:
        lines.append(f"The story so far: {journal}")
    narration = _last_of(values.get("messages"), AIMessage, name="dungeon_master")
    if narration:
        lines.append(f"Last from the DM: {narration.strip()}")
    return "\n".join(lines)
