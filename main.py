import sys
import traceback
import uuid

from langchain_core.messages import AIMessageChunk, HumanMessage

from src.graph.game_orchestrator import create_game_graph, create_sqlite_checkpointer
from src.graph.game_state import create_default_game_state

EXIT_COMMANDS = {"quit", "exit"}

# Nodes whose output is prose the player reads, so it is worth showing token by
# token. At 4.4 tok/s a finished narration is ~25 s away; the first token is ~3.5 s
# away. Everything else in the graph — routing decisions, the dice parse — emits
# tokens too, and none of it is for the player.
STREAMING_NODES = {"dungeon_master", "researcher"}

# A node can make more than one LLM call — the DM narrates, then extracts world
# state into JSON. Both carry the same node name, so the second is tagged.
INTERNAL_TAG = "internal"


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


def main() -> None:
    try:
        checkpointer = create_sqlite_checkpointer()
        game_graph = create_game_graph(checkpointer=checkpointer)
    except Exception as exc:
        print(f"Failed to create game graph: {exc}")
        traceback.print_exc()
        sys.exit(1)

    # One thread_id per session. Reuse a previous id to resume that campaign —
    # the checkpointer restores its full message history and game_state.
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("Initializing D&D adventure...")
    print(f"Session thread: {thread_id}")
    print("Type 'quit' or 'exit' to end the session.\n")

    # The default state seeds the first turn only. Afterwards the checkpointer
    # holds the state, and each turn passes just the new user message — the
    # add_messages reducer appends it to the stored history.
    pending_state = create_default_game_state()
    seeded = False

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

        turn = {
            "messages": [HumanMessage(content=user_input)],
            "current_task": user_input,
        }
        if not seeded:
            turn = {**pending_state, **turn}
            seeded = True

        # How many messages existed before this turn, so we can print only the
        # new ones. A thread with no checkpoint yet has no values at all.
        try:
            snapshot = game_graph.get_state(config)
            before = len(snapshot.values.get("messages", [])) if snapshot.values else 0
        except Exception:
            before = 0

        try:
            streamed = _run_turn(game_graph, turn, config)
            messages = game_graph.get_state(config).values.get("messages", [])
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


if __name__ == "__main__":
    main()
