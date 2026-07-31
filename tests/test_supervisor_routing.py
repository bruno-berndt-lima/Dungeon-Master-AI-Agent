"""Routing tests for `GameSupervisor`.

No model daemon is used. The pre-filter is a pure function, and the LLM path is
exercised by replacing `supervisor.llm` with a stub — so this runs in CI and
pins the *contract* (what the supervisor does with a given router answer) rather
than the model's judgement, which is not a thing tests can assert on.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.supervisor import (
    AGENT_TYPES,
    ROUTING_OPTIONS,
    GameSupervisor,
    prefilter_route,
)

pytestmark = pytest.mark.integration  # constructing the agent imports the stack


# --- the deterministic pre-filter (pure) ------------------------------------

@pytest.mark.parametrize(
    "request_text",
    [
        "roll a d20",
        "roll 2d10 + 1d6 for damage",
        "Roll 1d8",
        "2d6+1d8",
        "d20",
        "d20 + 3",
        "rolling 4d6 for stats",
    ],
)
def test_prefilter_catches_unambiguous_rolls(request_text):
    assert prefilter_route(request_text) == "dice_roller"


@pytest.mark.parametrize(
    "request_text",
    [
        # Dice notation, but a rules question — the classic false positive.
        "what does d20 mean",
        "how much is 2d6 on average",
        "does a d20 roll of 1 always miss",
        "explain why 2d6 beats 1d12",
        # Notation as description, not a request to roll.
        "my sword does 2d6 slashing damage",
        # No notation at all.
        "I open the door",
        "how does sneak attack work",
        "roll with it",          # roll verb, no notation
        "",
        "   ",
    ],
)
def test_prefilter_defers_to_the_model_when_unsure(request_text):
    """A wrong fast answer is worse than a slow right one."""
    assert prefilter_route(request_text) is None


def test_prefilter_returns_only_real_agent_types():
    assert prefilter_route("roll a d20") in AGENT_TYPES


# --- the LLM path -----------------------------------------------------------

class StubRouter:
    """Stands in for the structured-output runnable."""

    def __init__(self, result):
        self.result = result
        self.calls = []

    def invoke(self, messages, *args, **kwargs):
        self.calls.append(messages)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def make_supervisor(result):
    supervisor = GameSupervisor()
    stub = StubRouter(result)
    supervisor.llm = stub
    return supervisor, stub


def state(current_task="", messages=None):
    return {"current_task": current_task, "messages": messages or []}


@pytest.mark.parametrize("agent", AGENT_TYPES)
def test_llm_decision_becomes_the_destination(agent):
    supervisor, _ = make_supervisor({"next": agent})
    command = supervisor.process_task(state("something ambiguous"))
    assert command.goto == agent
    assert command.update["active_agent"] == agent


def test_finish_terminates_the_graph_but_still_says_something():
    """A turn that ends with no message at all is indistinguishable from a hang."""
    supervisor, _ = make_supervisor({"next": "FINISH"})
    command = supervisor.process_task(state("thanks, that's all"))
    assert command.goto == "__end__"
    assert command.update["messages"][0].content.strip()


def test_prefilter_short_circuits_before_the_model():
    supervisor, stub = make_supervisor({"next": "researcher"})
    command = supervisor.process_task(state("roll a d20"))
    assert command.goto == "dice_roller"
    assert stub.calls == [], "the model was called for an unambiguous dice request"


# --- no silent fallback (KNOWN_ISSUES #5) -----------------------------------

@pytest.mark.parametrize(
    "bad_result",
    [
        {"next": "librarian"},          # not a real node
        {"next": None},
        {},                             # schema key missing
        "researcher",                   # not a dict
        None,
        RuntimeError("daemon is down"),
    ],
)
def test_router_failures_end_the_turn_instead_of_guessing(bad_result):
    """The old code sent every failure to `researcher`, which then generated a
    confident answer to a question the player never asked."""
    supervisor, _ = make_supervisor(bad_result)
    command = supervisor.process_task(state("some input"))

    assert command.goto == "__end__"
    assert command.goto != "researcher"

    message = command.update["messages"][0]
    assert isinstance(message, AIMessage)
    assert "could not" in message.content.lower()


# --- routing input (KNOWN_ISSUES #6) ----------------------------------------

def test_routes_on_the_current_task_not_the_message_tail():
    """After a roll, the newest message is the result. Routing on it is #6."""
    supervisor, stub = make_supervisor({"next": "researcher"})
    supervisor.process_task(
        state(
            current_task="how does sneak attack work",
            messages=[
                HumanMessage(content="how does sneak attack work"),
                AIMessage(content="🎲 Rolled 2d10: **14**", name="dice_roller"),
            ],
        )
    )
    routed_on = stub.calls[0][-1].content
    assert routed_on == "how does sneak attack work"
    assert "Rolled" not in routed_on


def test_falls_back_to_the_latest_human_message():
    supervisor, stub = make_supervisor({"next": "researcher"})
    supervisor.process_task(
        state(
            current_task="",
            messages=[
                HumanMessage(content="first question"),
                AIMessage(content="an answer", name="researcher"),
                HumanMessage(content="second question"),
            ],
        )
    )
    assert stub.calls[0][-1].content == "second question"


def test_routing_prompt_does_not_grow_with_history():
    """Routing input is constant-size, so latency stays flat across a campaign."""
    supervisor, stub = make_supervisor({"next": "researcher"})
    long_history = [HumanMessage(content=f"turn {i}") for i in range(200)]
    supervisor.process_task(state("what is a beholder", long_history))
    assert len(stub.calls[0]) == 2  # system prompt + the request


# --- logging ----------------------------------------------------------------

def test_both_routers_log_their_decision(tmp_path, monkeypatch):
    for request_text, expected_router in [("roll a d20", "prefilter"),
                                          ("what is a beholder", "llm")]:
        supervisor, _ = make_supervisor({"next": "researcher"})
        logged = {}
        monkeypatch.setattr(
            supervisor, "_log_interaction",
            lambda query, response, metadata=None, _l=logged: _l.update(metadata or {}),
        )
        supervisor.process_task(state(request_text))
        assert logged["router"] == expected_router
        assert logged["routed_to"] in ROUTING_OPTIONS + ["__end__"]
