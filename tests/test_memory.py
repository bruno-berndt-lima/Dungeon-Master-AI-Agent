"""`src/agents/memory.py` — the rolling journal, and the sixty-turn session
the spec asks for: a bounded prompt, and a name from turn 2 still known at
turn 60.
"""

import random
import re

import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage  # noqa: E402

from src.agents.dungeon_master import DungeonMaster  # noqa: E402
from src.agents.memory import KEEP_RECENT, MAX_MESSAGES, Memory, transcript  # noqa: E402
from src.engine.pregens import pregen  # noqa: E402
from src.graph.game_orchestrator import create_game_graph  # noqa: E402
from src.graph.game_state import create_default_game_state, put_party  # noqa: E402


class StubLLM:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []
        self.configs = []

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, config=None):
        self.prompts.append(list(messages))
        self.configs.append(config)
        if not self.replies:
            return AIMessage(content="The dust settles.")
        reply = self.replies.pop(0)
        return reply if isinstance(reply, AIMessage) else AIMessage(content=reply)


class JournalStub(StubLLM):
    """A stand-in journal keeper: carries forward every sentence that names
    someone (a capitalised word), deduplicated. What a real summariser is
    asked to do, done mechanically so the pipeline can be checked."""

    def invoke(self, messages, config=None):
        self.prompts.append(list(messages))
        self.configs.append(config)
        text = messages[-1].content
        journal, _, new = text.partition("What happened next:")
        journal = journal.replace("Journal so far:", "").replace("(nothing yet)", "").strip()
        kept = [s.strip() for s in journal.split(".") if s.strip()]
        for line in new.splitlines():
            body = line.split(":", 1)[1].strip() if ":" in line else line.strip()
            for sentence in body.split("."):
                sentence = sentence.strip()
                if sentence and re.search(r"\b[A-Z][a-z]+\b", sentence) and sentence not in kept:
                    kept.append(sentence)
        return AIMessage(content=". ".join(kept) + ("." if kept else ""))


def messages_of(n, start=0):
    out = []
    for i in range(start, start + n):
        out.append(HumanMessage(content=f"player line {i}", id=f"h{i}"))
        out.append(AIMessage(content=f"dm line {i}", name="dungeon_master", id=f"a{i}"))
    return out


def state_with(messages, summary=""):
    s = dict(create_default_game_state())
    s["messages"] = messages
    s["summary"] = summary
    return s


# --- the node on its own ----------------------------------------------------------------

def test_below_the_threshold_nothing_happens():
    stub = StubLLM("should not be called")
    node = Memory(llm=stub)
    cmd = node.process_task(state_with(messages_of(MAX_MESSAGES // 2)))
    assert cmd.update is None and stub.prompts == []


def test_above_the_threshold_the_oldest_are_folded_and_removed():
    stub = StubLLM("The journal, updated.")
    node = Memory(llm=stub)
    history = messages_of(MAX_MESSAGES // 2 + 4)  # 2 per turn → past the threshold
    cmd = node.process_task(state_with(history, summary="Old journal."))
    assert cmd.update["summary"] == "The journal, updated."
    removed = [m for m in cmd.update["messages"] if isinstance(m, RemoveMessage)]
    assert [m.id for m in removed] == [m.id for m in history[: len(history) - KEEP_RECENT]]
    assert len(history) - len(removed) == KEEP_RECENT


def test_the_journal_keeper_is_given_the_old_journal_and_the_transcript_and_is_internal():
    stub = StubLLM("new")
    node = Memory(llm=stub)
    history = messages_of(MAX_MESSAGES // 2 + 4)
    node.process_task(state_with(history, summary="Kara owes the innkeeper a favour."))
    prompt = stub.prompts[0][-1].content
    assert "Journal so far:\nKara owes the innkeeper a favour." in prompt
    assert "Player: player line 0" in prompt and "DM: dm line 0" in prompt
    assert f"player line {MAX_MESSAGES // 2 + 3}" not in prompt  # the recent tail is not folded
    assert stub.configs[0] == {"tags": ["internal"]}


def test_a_failed_fold_removes_nothing():
    class Broken(StubLLM):
        def invoke(self, messages, config=None):
            raise RuntimeError("daemon down")

    cmd = Memory(llm=Broken()).process_task(state_with(messages_of(MAX_MESSAGES)))
    assert cmd.update is None


def test_an_empty_reply_keeps_the_old_journal():
    cmd = Memory(llm=StubLLM(AIMessage(content=""))).process_task(state_with(messages_of(MAX_MESSAGES), summary="kept"))
    assert cmd.update["summary"] == "kept"


def test_transcript_renders_players_dm_and_table_notes_and_skips_bare_tool_calls():
    text = transcript([
        HumanMessage(content="I open the door", name="bruno"),
        AIMessage(content="", tool_calls=[{"name": "get_scene", "args": {}, "id": "1", "type": "tool_call"}]),
        ToolMessage(content="Not in combat.", tool_call_id="1", name="get_scene"),
        AIMessage(content="It creaks.", name="dungeon_master"),
        AIMessage(content="🎲 12", name="intake"),
    ])
    assert text == "bruno: I open the door\n[table] Not in combat.\nDM: It creaks.\nintake: 🎲 12"


# --- through the graph: the sixty-turn session ---------------------------------------------

BUDGET_CHARS = 6000  # ~1,500 tokens for the whole DM prompt, system prompt included


def test_a_sixty_turn_session_stays_within_budget_and_remembers_turn_two():
    dm = StubLLM()
    journal = JournalStub()
    g = create_game_graph(dm_llm=dm, memory_llm=journal, rng=random.Random(0))

    values = dict(create_default_game_state())
    values["party"] = put_party({"Kara": pregen("rogue", "p1", "Kara")})
    lines = ["I ask the innkeeper his name; he says he is called Marrow and pours an ale."] + [
        f"I look around the market for the {i}th time and chat with a trader." for i in range(3, 61)
    ]
    dm.replies = ["The innkeeper, Marrow, smiles and pours."] + [f"The trader nods. Turn {i} passes." for i in range(3, 61)]

    largest_prompt = 0
    history = []
    for i, text in enumerate(lines, start=2):
        before = len(dm.prompts)
        values = g.invoke({**values, "messages": history + [HumanMessage(content=text)], "current_task": text})
        history = list(values["messages"])
        prompt_chars = sum(len(str(m.content)) for m in dm.prompts[before])
        largest_prompt = max(largest_prompt, prompt_chars)

    assert len(values["messages"]) <= MAX_MESSAGES
    assert largest_prompt <= BUDGET_CHARS, largest_prompt
    assert "Marrow" in values["summary"]
    assert journal.prompts, "the journal keeper was never called"
    # The DM's last prompt carried the journal, so turn 2's fact was in front of it at turn 60.
    briefing = "\n".join(str(m.content) for m in dm.prompts[-1])
    assert "Marrow" in briefing and "The story so far" in briefing


def test_the_dm_reads_the_journal_as_the_story_so_far():
    stub = StubLLM("Fine.")
    dm = DungeonMaster(llm=stub)
    s = dict(create_default_game_state())
    s["summary"] = "Kara owes Marrow a favour."
    s["messages"] = [HumanMessage(content="hello")]
    dm.process_task(s)
    briefing = next(m for m in stub.prompts[0] if getattr(m, "name", None) == "table")
    assert "## The story so far\n\nKara owes Marrow a favour." in briefing.content
    assert "Kara owes Marrow" not in stub.prompts[0][0].content  # never in the cacheable prefix
