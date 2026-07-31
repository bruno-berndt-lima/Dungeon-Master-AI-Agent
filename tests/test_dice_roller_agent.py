"""Contract tests for `DiceRollerAgent`.

The model only parses; `DiceRoller` performs every roll. That split is what
makes these tests possible without a daemon — the parser is stubbed, and the
arithmetic underneath is real.
"""

import re

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.dice_roller import (
    DiceParseError,
    DiceRollerAgent,
    extract_dice_expression,
    extract_roll_flags,
)

pytestmark = pytest.mark.integration  # constructing the agent imports the stack


class StubParser:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def invoke(self, messages, *args, **kwargs):
        self.calls.append(messages)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def make_agent(parsed):
    agent = DiceRollerAgent()
    agent.parser = StubParser(parsed)
    return agent


def parsed(notation, modifier=0, advantage=False, disadvantage=False, description=""):
    return {
        "dice_notation": notation,
        "modifier": modifier,
        "has_advantage": advantage,
        "has_disadvantage": disadvantage,
        "description": description,
    }


def state(task="roll 2d6"):
    return {"current_task": task, "messages": [HumanMessage(content=task)]}


def total_of(message: str) -> int:
    return int(re.search(r"\*\*(-?\d+)\*\*", message).group(1))


# --- the roll is real -------------------------------------------------------
#
# These drive the agent through the request text, because that is what decides
# the roll now. The stub is present only to prove the model is not consulted.

def test_the_total_is_within_bounds_and_matches_the_dice():
    agent = make_agent(parsed("9d9", modifier=99))
    result = agent.process_task(state("roll 2d6+3 for damage")).update["messages"][0].content
    assert 5 <= total_of(result) <= 15   # 2..12 plus 3
    assert "damage" in result


def test_a_negative_modifier_is_subtracted():
    agent = make_agent(parsed("9d9"))
    total = total_of(agent.process_task(state("roll 2d6-1")).update["messages"][0].content)
    assert 1 <= total <= 11


def test_multiple_dice_types_are_all_rolled():
    agent = make_agent(parsed("9d9"))
    result = agent.process_task(state("roll 2d8 + 1d6")).update["messages"][0].content
    assert 3 <= total_of(result) <= 22
    assert "2d8" in result and "1d6" in result


def test_advantage_takes_the_higher_of_two_rolls():
    agent = make_agent(parsed("9d9"))
    result = agent.process_task(
        state("roll 1d20 with advantage")).update["messages"][0].content
    rolls = [int(n) for n in re.search(r"rolls: (\d+) and (\d+)", result).groups()]
    assert total_of(result) == max(rolls)
    assert "advantage" in result


def test_disadvantage_takes_the_lower_of_two_rolls():
    agent = make_agent(parsed("9d9"))
    result = agent.process_task(
        state("roll 1d20 with disadvantage")).update["messages"][0].content
    rolls = [int(n) for n in re.search(r"rolls: (\d+) and (\d+)", result).groups()]
    assert total_of(result) == min(rolls)
    assert "disadvantage" in result


@pytest.mark.parametrize("request_text,low,high", [
    ("roll 1d20", 1, 20),
    ("roll d20", 1, 20),
    ("roll 4d6", 4, 24),
    ("roll 2d6-1", 1, 11),
    ("roll 2d6+3", 5, 15),
])
def test_bounds_hold_across_notations(request_text, low, high):
    agent = make_agent(parsed("9d9", modifier=99))
    for _ in range(20):
        assert low <= total_of(
            agent.process_task(state(request_text)).update["messages"][0].content
        ) <= high


# --- deterministic extraction (pure, no model) ------------------------------

@pytest.mark.parametrize("request_text,notation,modifier", [
    ("roll 2d6-1", "2d6", -1),
    ("roll 1d20+5 for attack", "1d20", 5),
    ("roll 2d8 + 1d6 for damage", "2d8+1d6", 0),
    ("roll a d20 with advantage for stealth", "1d20", 0),
    ("roll 1d20 with disadvantage", "1d20", 0),
    ("roll 4d6", "4d6", 0),
    ("roll 3d8+2", "3d8", 2),
    ("2d6+1d8+3", "2d6+1d8", 3),
    ("roll d20", "1d20", 0),               # normalised
    ("roll 2d6+3-1", "2d6", 2),            # modifiers accumulate
    ("roll for initiative", None, 0),      # no notation — the model's job
    ("what is a beholder", None, 0),
])
def test_extraction_reads_only_what_is_written(request_text, notation, modifier):
    """Every one of these was measured wrong when a 3B model was asked instead:
    it invented +1 on "2d8 + 1d6", +5 on a plain advantage roll, +2 on
    "roll for initiative". An invented modifier silently changes the result."""
    assert extract_dice_expression(request_text) == (notation, modifier)


@pytest.mark.parametrize("request_text,advantage,disadvantage", [
    ("roll a d20 with advantage", True, False),
    ("roll 1d20 with disadvantage", False, True),
    ("roll 2d6", False, False),
    # "disadvantage" contains "advantage" — the classic substring trap.
    ("disadvantage on this one", False, True),
])
def test_advantage_flags_are_keyword_exact(request_text, advantage, disadvantage):
    assert extract_roll_flags(request_text)[:2] == (advantage, disadvantage)


@pytest.mark.parametrize("request_text,description", [
    ("roll 1d20 for attack", "attack"),
    ("roll 2d6 to check stealth", "stealth"),
    ("roll 1d20", ""),
    ("roll 1d20 with advantage", ""),
])
def test_purpose_is_read_from_the_request(request_text, description):
    assert extract_roll_flags(request_text)[2] == description


def test_subtracted_dice_are_rejected_at_extraction():
    with pytest.raises(DiceParseError, match="cannot subtract dice"):
        extract_dice_expression("roll 2d6-1d4")


def test_written_notation_never_reaches_the_model():
    """The common path costs no LLM call at all — measured 4.6 s before."""
    agent = make_agent(parsed("9d9", modifier=99))  # would be wrong if consulted
    result = agent.process_task(state("roll 2d6+3")).update["messages"][0].content
    assert agent.parser.calls == []
    assert "2d6" in result
    assert 5 <= total_of(result) <= 15


def test_the_model_is_consulted_only_when_no_dice_are_named():
    agent = make_agent(parsed("1d20", description="initiative"))
    result = agent.process_task(state("roll for initiative")).update["messages"][0].content
    assert len(agent.parser.calls) == 1
    assert 1 <= total_of(result) <= 20


@pytest.mark.parametrize("invented", [
    {"dice_notation": "1d20", "modifier": 2},    # in the modifier field
    {"dice_notation": "1d20+2", "modifier": 0},  # folded into the notation
])
def test_a_modifier_the_model_invents_is_ignored_when_the_text_has_none(invented):
    """Both were observed live. `llama3.2:3b` answered "roll for initiative"
    with `1d20+2` — a +2 the player never asked for, which would have been
    added to every initiative roll in the campaign."""
    agent = make_agent(invented)
    for _ in range(20):
        total = total_of(
            agent.process_task(state("roll for initiative")).update["messages"][0].content
        )
        assert 1 <= total <= 20, "an invented modifier reached the total"


def test_a_modifier_the_player_did_write_survives_the_fallback():
    """The rule is "only what the request says" — not "never a modifier"."""
    agent = make_agent(parsed("1d20"))
    for _ in range(20):
        total = total_of(
            agent.process_task(state("roll for initiative +2")).update["messages"][0].content
        )
        assert 3 <= total <= 22


# --- the parser is no longer defended against -------------------------------

def test_the_defensive_stack_is_gone_from_the_parse_code():
    """PR-05 deleted ~70 lines that existed only to survive unreliable JSON.

    Checks executable code, not docstrings — the docstring deliberately names
    what was removed.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(DiceRollerAgent._parse_dice_request).strip())
    function = tree.body[0]
    if ast.get_docstring(function):
        function.body = function.body[1:]
    code = ast.unparse(function)

    assert "PQXYpqxy" not in code, "the placeholder guard should be gone"
    assert "json.loads" not in code, "manual JSON parsing should be gone"
    assert "re.sub" not in code, "the markdown-fence stripper should be gone"
    assert "to check" not in code, "the description salvage should be gone"
    assert "1d20" not in code, "the silent 1d20 default should be gone"


def test_the_prompt_lives_in_the_prompts_module():
    """CLAUDE.md convention: no inlined system prompts in agent classes."""
    import inspect
    from src.agents import dice_roller
    from src.prompts.prompts import DICE_PARSE_PROMPT

    assert "EXAMPLES:" not in inspect.getsource(dice_roller)
    assert DICE_PARSE_PROMPT.strip()


def test_the_parsed_request_reaches_the_log(monkeypatch):
    agent = make_agent(parsed("9d9"))
    logged = []
    monkeypatch.setattr(agent, "_log_interaction",
                        lambda **kwargs: logged.append(kwargs))
    agent.process_task(state("roll 2d6+3 for damage"))
    roll = logged[0]["metadata"]["dice_roll"]
    assert roll["dice_notation"] == "2d6"
    assert roll["modifier"] == 3
    assert roll["description"] == "damage"


# --- failure is reported, not guessed ---------------------------------------

@pytest.mark.parametrize("bad", [
    {"dice_notation": "", "modifier": 0},
    {"dice_notation": "not dice", "modifier": 0},
    {"dice_notation": "2d6-1d4", "modifier": 0},   # unrepresentable
    {},
    None,
    "2d6",
    RuntimeError("daemon is down"),
])
def test_an_unparseable_request_says_so_instead_of_rolling_1d20(bad):
    """The old code returned ("1d20", 0, False, False, "dice roll") on any
    failure — a confident number for a roll nobody asked for.

    The request names no dice, so this exercises the model fallback path.
    """
    agent = make_agent(bad)
    command = agent.process_task(state("roll for initiative"))
    content = command.update["messages"][0].content

    assert command.goto == "__end__"
    assert "could not tell what to roll" in content
    assert "**" not in content, "no total should be reported"


def test_parse_errors_are_raised_not_swallowed():
    agent = make_agent({"dice_notation": "nonsense", "modifier": 0})
    with pytest.raises(DiceParseError):
        agent._parse_dice_request("roll something")


def test_the_node_always_returns_a_command_with_a_message():
    for bad in [RuntimeError("boom"), None, {"dice_notation": ""}]:
        command = make_agent(bad).process_task(state("roll for initiative"))
        assert command.goto == "__end__"
        message = command.update["messages"][0]
        assert isinstance(message, AIMessage)
        assert message.name == "dice_roller"
        assert message.content.strip()
