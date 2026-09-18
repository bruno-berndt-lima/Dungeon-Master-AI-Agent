"""The REPL. One campaign per run, resumable.

    python main.py                       # start a new campaign
    python main.py --thread 3f9a1c2e     # pick one up where it left off
    python main.py --thread tuesday      # any name works; new if unseen
    python main.py --list                # what is in the checkpoint database

Campaign state lives in the SQLite checkpointer (`game_state.db`), keyed by
thread id. The REPL is also the development harness for the Discord bot
(ROADMAP PR-22), where the channel id plays the role of `--thread`.
"""

import argparse
import sys
import traceback
from typing import Optional, Sequence

from langchain_core.messages import AIMessageChunk, HumanMessage

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

# A node can make more than one LLM call — the DM narrates, then extracts world
# state into JSON. Both carry the same node name, so the second is tagged.
INTERNAL_TAG = "internal"


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
    lines = [f"{'thread':{width}}  {'last active':16}  {'turns':>5}  where / last input"]
    for c in campaigns:
        when = c.last_active[:16].replace("T", " ")
        tail = c.location or c.last_prompt
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


def _run_turn(game_graph, turn, config) -> dict:
    """Streams a turn, printing prose as it arrives.

    Returns {agent name: text already printed live}, so the caller can skip
    reprinting it — and can still show anything the node appended after the
    model finished, such as the researcher's list of sources.
    """
    streamed = {}

    for chunk, metadata in game_graph.stream(
        turn, config=config, stream_mode="messages"
    ):
        # Two kinds of thing arrive here: AIMessageChunk for each token, and the
        # finished AIMessage the node writes to state. Printing both shows every
        # narration twice.
        if not isinstance(chunk, AIMessageChunk):
            continue

        node = metadata.get("langgraph_node")
        if node not in STREAMING_NODES:
            continue
        if INTERNAL_TAG in (metadata.get("tags") or ()):
            continue

        text = getattr(chunk, "content", "")
        if not text:
            continue

        if node not in streamed:
            print(f"\n[{node}] ", end="", flush=True)
            streamed[node] = ""
        streamed[node] += text
        print(text, end="", flush=True)

    return streamed


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
    else:
        print(f"Resuming campaign {thread_id}.")
        print(recap(values))
    print("Type 'quit' or 'exit' to end the session.\n")

    while True:
        try:
            user_input = input("Ask a D&D question: ").strip()
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
            streamed = _run_turn(game_graph, turn, config)
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

            name = getattr(message, "name", None)
            content = getattr(message, "content", "")

            if name in streamed:
                # A node may add to its answer after the model stops — the
                # researcher appends the passages it used. Show the tail rather
                # than reprinting the whole thing.
                already = streamed[name]
                tail = content[len(already):] if content.startswith(already) else ""
                if tail.strip():
                    print(tail, end="", flush=True)
                print("\n")
                continue

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
