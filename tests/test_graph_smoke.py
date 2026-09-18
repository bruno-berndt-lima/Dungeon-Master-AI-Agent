"""Smoke tests for graph construction.

Marked `integration` because these require the full dependency stack (langgraph,
langchain-core, chromadb, ...) and Python 3.11+. They do NOT require a running
model daemon — agent constructors build clients lazily and ResearcherAgent
swallows vectorstore failures.

Run everything except these with:   pytest -m "not integration"
"""

import sys

import pytest

pytestmark = pytest.mark.integration

if sys.version_info < (3, 11):
    pytest.skip(
        "the graph modules use PEP 646 syntax, "
        "which does not parse before Python 3.11",
        allow_module_level=True,
    )

pytest.importorskip("langgraph", reason="full dependency stack not installed")

from src.graph.game_orchestrator import create_game_graph  # noqa: E402
from src.graph.game_state import create_default_game_state  # noqa: E402

EXPECTED_NODES = {"intake", "dungeon_master", "tools", "memory", "researcher"}


def test_graph_compiles():
    assert create_game_graph() is not None


def test_graph_registers_every_agent_node():
    nodes = set(create_game_graph().get_graph().nodes)
    missing = EXPECTED_NODES - nodes
    assert not missing, f"nodes missing from the compiled graph: {sorted(missing)}"


def test_default_state_declares_every_field():
    """Guards the GameState contract that PR-03 will rewrite."""
    state = create_default_game_state()
    expected = {
        "messages", "current_task", "tool_steps",
        "party", "encounter", "pending", "summary", "last_response",
    }
    assert set(state) == expected


def test_the_never_written_fields_are_gone():
    """KNOWN_ISSUES #16: players, npcs, current_speaker, turn_order and
    requires_player_input were declared and never assigned. PR-16 replaced
    them with the engine's party / encounter / pending."""
    state = create_default_game_state()
    for stale in ("players", "npcs", "current_speaker", "turn_order", "requires_player_input"):
        assert stale not in state


def test_the_actors_package_is_gone():
    """KNOWN_ISSUES #14: src/actors was type-hint-only. The engine replaces it."""
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.actors")


def test_next_agent_is_gone():
    """Routing belongs to Command(goto=...). A next_agent mirror could disagree
    with it, and main.py used to exit the REPL whenever it read "FINISH" —
    which ResearcherAgent set on every successful answer."""
    assert "next_agent" not in create_default_game_state()


def test_the_supervisor_and_dice_agents_are_gone():
    """PR-18: routing is code (`intake`), dice are the engine's. No module,
    no prompt, no model row remains for either."""
    import importlib

    from src.models.llm import AGENT_MODELS
    from src.prompts import prompts

    for module in ("src.agents.supervisor", "src.agents.dice_roller"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module)
    assert set(AGENT_MODELS) == {"researcher", "dungeon_master", "memory"}
    assert not hasattr(prompts, "SUPERVISOR_PROMPT") and not hasattr(prompts, "SCENE_EXTRACTION_PROMPT")
