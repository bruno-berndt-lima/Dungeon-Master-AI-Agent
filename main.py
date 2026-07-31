import sys
import traceback
import uuid

from langchain_core.messages import HumanMessage

from src.graph.game_orchestrator import create_game_graph, create_sqlite_checkpointer
from src.graph.game_state import create_default_game_state

EXIT_COMMANDS = {"quit", "exit"}


def _render(message) -> None:
    """Prints one assistant message with its agent name, if it carries one."""
    name = getattr(message, "name", None)
    content = getattr(message, "content", str(message))
    print(f"\n[{name or 'assistant'}] {content}\n")


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
            result = game_graph.invoke(turn, config=config)
        except Exception as exc:
            print(f"\nAn error occurred: {exc}")
            traceback.print_exc()
            continue

        # Show only what this turn produced. The session continues until the
        # user asks to leave — no agent decides that on their behalf.
        for message in result.get("messages", [])[before:]:
            if not isinstance(message, HumanMessage):
                _render(message)


if __name__ == "__main__":
    main()
