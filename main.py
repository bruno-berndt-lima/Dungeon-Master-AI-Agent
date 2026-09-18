"""The REPL. One campaign per run, resumable.

    python main.py                       # start a new campaign
    python main.py --thread 3f9a1c2e     # pick one up where it left off
    python main.py --thread tuesday      # any name works; new if unseen
    python main.py --list                # what is in the checkpoint database
    DND_SHOW_TOOLS=1 python main.py      # also print every tool call and its answer

Campaign state lives in the SQLite checkpointer (`game_state.db`), keyed by
thread id. The REPL is also the development harness for the Discord bot
(ROADMAP PR-22), where the channel id plays the role of `--thread`.
"""

import argparse
import os
import sys
import traceback
from typing import Optional, Sequence

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from src.agents.dungeon_master import NARRATION_TAG
from src.graph.campaigns import (
    is_new_campaign,
    list_campaigns,
    new_thread_id,
    recap,
    seed_turn,
)
from src.graph.game_orchestrator import (
    DEFAULT_CHECKPOINT_DB,
    create_game_graph,
    create_sqlite_checkpointer,
)

EXIT_COMMANDS = {"quit", "exit"}

# Nodes whose output is prose the player reads, so it is worth showing token by
# token. At 4.4 tok/s a finished narration is ~25 s away; the first token is ~3.5 s
# away. Everything else in the graph — routing decisions, the dice parse — emits
# tokens too, and none of it is for the player.
STREAMING_NODES = {"dungeon_master", "researcher"}

# A node can make more than one LLM call — the researcher rewrites a query
# before answering. Both carry the same node name, so the internal one is tagged.
INTERNAL_TAG = "internal"

# Tool calls and their answers are machinery. Set DND_SHOW_TOOLS=1 to watch them.
SHOW_TOOLS = bool(os.environ.get("DND_SHOW_TOOLS", "").strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play a D&D 5e campaign with the AI Dungeon Master.")
    parser.add_argument(
        "--thread",
        metavar="ID",
        help="campaign to resume, or a name for a new one. Omit to start a fresh campaign.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list the campaigns in the checkpoint database and exit.",
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_CHECKPOINT_DB,
        help=f"checkpoint database (default: {DEFAULT_CHECKPOINT_DB}).",
    )
    return parser


def format_campaigns(campaigns) -> str:
    if not campaigns:
        return "No campaigns yet. Start one with: python main.py"
    width = max(len("thread"), *(len(c.thread_id) for c in campaigns))
    lines = [f"{'thread':{width}}  {'last active':16}  {'turns':>5}  party / last input"]
    for c in campaigns:
        when = c.last_active[:16].replace("T", " ")
        tail = ", ".join(c.party) or c.last_prompt
        if len(tail) > 50:
            tail = tail[:47] + "..."
        lines.append(f"{c.thread_id:{width}}  {when:16}  {c.turns:>5}  {tail}")
    lines.append("\nResume one with: python main.py --thread <thread>")
    return "\n".join(lines)


def _render(message) -> None:
    """Prints one assistant message with its agent name, if it carries one."""
    name = getattr(message, "name", None)
    content = getattr(message, "content", str(message))
    print(f"\n[{name or 'assistant'}] {content}\n")


def _show_tool_traffic(message) -> None:
    """The call and its answer, when DND_SHOW_TOOLS is set."""
    if isinstance(message, ToolMessage):
        args = (message.additional_kwargs or {}).get("args") or {}
        shown = ", ".join(f"{k}={v!r}" for k, v in args.items())
        answer = str(message.content).replace("\n", "\n      ")
        print(f"\n  ⚙ {message.name}({shown})\n      {answer}")


def _run_turn(game_graph, turn, config) -> dict:
    """Streams a turn. Returns {node: text shown for its latest model call}.

    Two kinds of model call reach the screen differently:

    - **Streamed live** — the researcher, and any DM call tagged `narration`
      (narrate-only mode): it is known to be prose before it starts.
    - **Buffered** — the DM's planning calls. A reply that ends in a tool call
      may begin with prose that assumes the tool's outcome; that text is not
      narration and must never show. So the tokens are held until the call
      ends, and shown only if no tool call arrived. On the M4 this is a
      moment; on the Intel CPU it is a pause. KNOWN_ISSUES #29.
    """
    streamed = {}
    shown = set()  # ids of messages already rendered whole, live
    buffers = {}  # node -> {"id": call id, "text": ..., "tools": bool}

    def flush(node):
        held = buffers.pop(node, None)
        if held and not held["tools"] and held["text"].strip():
            print(f"\n[{node}] {held['text'].strip()}", end="", flush=True)
            streamed[node] = held["text"].strip()

    for chunk, metadata in game_graph.stream(
        turn, config=config, stream_mode="messages"
    ):
        node = metadata.get("langgraph_node")
        if isinstance(chunk, ToolMessage):
            flush("dungeon_master")
            if SHOW_TOOLS:
                _show_tool_traffic(chunk)
            continue
        # A finished message from a node that does not stream (intake's dice
        # result, a resolved attack): show it now, in order, and remember it.
        if isinstance(chunk, AIMessage) and not isinstance(chunk, AIMessageChunk):
            if node not in STREAMING_NODES and not getattr(chunk, "tool_calls", None):
                flush("dungeon_master")
                _render(chunk)
                if getattr(chunk, "id", None):
                    shown.add(chunk.id)
            continue
        if not isinstance(chunk, AIMessageChunk):
            continue

        if node not in STREAMING_NODES:
            continue
        tags = metadata.get("tags") or ()
        if INTERNAL_TAG in tags:
            continue

        text = getattr(chunk, "content", "") or ""
        live = node != "dungeon_master" or NARRATION_TAG in tags

        if not live:
            held = buffers.get(node)
            if held is None or held["id"] != chunk.id:
                flush(node)
                held = buffers[node] = {"id": chunk.id, "text": "", "tools": False}
            if getattr(chunk, "tool_call_chunks", None):
                held["tools"] = True
            held["text"] += text
            continue

        if not text:
            continue
        if streamed.get(node) is None or buffers.pop(node, None) is not None or streamed.get(("id", node)) != chunk.id:
            print(f"\n[{node}] ", end="", flush=True)
            streamed[node] = ""
            streamed[("id", node)] = chunk.id
        streamed[node] += text
        print(text, end="", flush=True)

    for node in list(buffers):
        flush(node)
    return {"streamed": {k: v for k, v in streamed.items() if isinstance(k, str)}, "shown": shown}


def _state_values(game_graph, config) -> dict:
    """The thread's current state, or {} for a thread that has none yet."""
    try:
        snapshot = game_graph.get_state(config)
    except Exception:
        return {}
    return dict(snapshot.values or {})


def run_repl(game_graph, thread_id: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}

    values = _state_values(game_graph, config)
    if is_new_campaign(values):
        print(f"New campaign. Resume it later with: python main.py --thread {thread_id}")
        print("Start with /join <fighter|rogue|cleric|wizard|ranger|barbarian> [as <name>], then say what you do.")
    else:
        print(f"Resuming campaign {thread_id}.")
        print(recap(values))
    print("/help lists the commands. Type 'quit' or 'exit' to end the session.\n")

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEnding D&D session. Farewell, adventurer!")
            break

        if not user_input:
            continue
        if user_input.lower() in EXIT_COMMANDS:
            print("Ending D&D session. Farewell, adventurer!")
            break

        # The first turn of a brand-new thread carries the default state under
        # the new message; every later turn — including the first turn of a
        # *resumed* session — carries only the message, and the checkpointer
        # supplies the rest. Re-seeding a resumed thread would wipe game_state.
        values = _state_values(game_graph, config)
        before = len(values.get("messages") or [])
        turn = seed_turn(
            {"messages": [HumanMessage(content=user_input)], "current_task": user_input},
            values,
        )

        try:
            live = _run_turn(game_graph, turn, config)
            streamed, shown = live["streamed"], live["shown"]
            messages = _state_values(game_graph, config).get("messages") or []
        except Exception as exc:
            print(f"\nAn error occurred: {exc}")
            traceback.print_exc()
            continue

        # Show only what this turn produced, and only what was not already
        # printed token by token. The session continues until the user asks to
        # leave — no agent decides that on their behalf.
        for message in messages[before:]:
            if isinstance(message, HumanMessage):
                continue
            # The DM's tool traffic was shown live (or hidden); never reprint it.
            if isinstance(message, ToolMessage):
                continue
            if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
                continue

            if getattr(message, "id", None) in shown:
                continue

            name = getattr(message, "name", None)
            content = str(getattr(message, "content", "")).strip()

            if name in streamed and content.startswith(streamed[name].strip()):
                # A node may add to its answer after the model stops — the
                # researcher appends the passages it used. Show the tail rather
                # than reprinting the whole thing.
                tail = content[len(streamed[name].strip()):]
                if tail.strip():
                    print(tail, end="", flush=True)
                print("\n")
                continue

            # Not streamed, or not what was streamed: show it whole. Never
            # swallow a message on the strength of a prefix mismatch.
            _render(message)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        checkpointer = create_sqlite_checkpointer(args.db)
    except Exception as exc:
        print(f"Could not open the checkpoint database {args.db!r}: {exc}")
        return 1

    if args.list:
        print(format_campaigns(list_campaigns(checkpointer)))
        return 0

    try:
        game_graph = create_game_graph(checkpointer=checkpointer)
    except Exception as exc:
        print(f"Failed to create game graph: {exc}")
        traceback.print_exc()
        return 1

    print("Initializing D&D adventure...")
    run_repl(game_graph, args.thread or new_thread_id())
    return 0


if __name__ == "__main__":
    sys.exit(main())
