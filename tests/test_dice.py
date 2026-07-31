"""Unit tests for src/utils/dice.py.

This module is pure (stdlib only, no LLM, no I/O), so these tests run offline on
any interpreter and form the baseline test gate for the refactor.
"""

import random

import pytest

from src.utils.dice import DiceRoll, DiceRoller


# --------------------------------------------------------------------------- #
# parse_dice_string
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "notation, expected",
    [
        ("1d20", [(1, 20)]),
        ("d20", [(1, 20)]),           # bare "d" defaults to quantity 1
        ("2d6", [(2, 6)]),
        ("2d6+1d8", [(2, 6), (1, 8)]),
        ("2d6 + 1d8", [(2, 6), (1, 8)]),   # whitespace is stripped
        ("2D6", [(2, 6)]),                 # case-insensitive
        ("1d20+2d6+3d4", [(1, 20), (2, 6), (3, 4)]),
    ],
)
def test_parse_dice_string(notation, expected):
    assert DiceRoller.parse_dice_string(notation) == expected


def test_parse_drops_bare_numeric_modifier():
    """Documents current behavior: a bare "+3" term is silently discarded.

    Modifiers are extracted separately by DiceRollerAgent and applied after the
    roll, so the parser never sees them in the normal path.
    """
    assert DiceRoller.parse_dice_string("2d6+3") == [(2, 6)]


def test_parse_negative_modifier():
    """KNOWN_ISSUES #9. Was a strict xfail from PR-01 until PR-05 fixed it.

    A flat modifier is not dice. `DiceRollerAgent` extracts it separately and
    adds it to the total, so the parser's job is to skip it — not to fail on it,
    which is what splitting only on '+' used to cause (`int("6-1")`).
    """
    assert DiceRoller.parse_dice_string("2d6-1") == [(2, 6)]


@pytest.mark.parametrize(
    "notation,expected",
    [
        ("1d20+5", [(1, 20)]),          # trailing modifier, either sign
        ("1d20-5", [(1, 20)]),
        ("2d6+3-1", [(2, 6)]),          # several modifiers
        ("1d20+2d6+3", [(1, 20), (2, 6)]),
        ("2D6+1D8", [(2, 6), (1, 8)]),
    ],
)
def test_parse_separates_modifiers_from_dice(notation, expected):
    assert DiceRoller.parse_dice_string(notation) == expected


@pytest.mark.parametrize(
    "notation",
    [
        "2d6-1d4",   # cannot be represented as (quantity, sides) tuples
        "0d6",       # no dice
        "2d0",       # a die with no faces
        "",
        "   ",
        "hello",
        "2x6",
        "+",
        "d",
    ],
)
def test_parse_rejects_what_it_cannot_represent(notation):
    """Returning a plausible roll for input we don't understand is how a roll
    silently becomes the wrong roll."""
    with pytest.raises(ValueError):
        DiceRoller.parse_dice_string(notation)


def test_subtracted_dice_are_rejected_rather_than_silently_added():
    """The old parser would have rolled "2d6-1d4" as 2d6+1d4."""
    with pytest.raises(ValueError, match="cannot subtract dice"):
        DiceRoller.parse_dice_string("2d6-1d4")


def test_roll_multiple_handles_a_negative_modifier_notation():
    """End to end: the notation that used to raise now rolls."""
    rolls = DiceRoller.roll_multiple("2d6-1")
    assert len(rolls) == 1
    assert rolls[0].dice_type == 6
    assert 2 <= rolls[0].total <= 12


# --------------------------------------------------------------------------- #
# roll_single_type
# --------------------------------------------------------------------------- #

def test_roll_single_type_shape():
    roll = DiceRoller.roll_single_type(4, 6)
    assert isinstance(roll, DiceRoll)
    assert roll.dice_type == 6
    assert len(roll.results) == 4
    assert roll.total == sum(roll.results)


def test_roll_single_type_stays_in_bounds():
    for _ in range(200):
        roll = DiceRoller.roll_single_type(3, 8)
        assert all(1 <= r <= 8 for r in roll.results)


def test_roll_single_type_zero_quantity():
    roll = DiceRoller.roll_single_type(0, 20)
    assert roll.results == []
    assert roll.total == 0


def test_roll_is_deterministic_under_seed():
    """Same seed, same sequence — no assertion on specific values, which would
    be brittle across interpreter versions."""
    random.seed(1234)
    first = DiceRoller.roll_multiple("2d6+1d8")
    random.seed(1234)
    second = DiceRoller.roll_multiple("2d6+1d8")
    assert [r.results for r in first] == [r.results for r in second]


# --------------------------------------------------------------------------- #
# roll_multiple
# --------------------------------------------------------------------------- #

def test_roll_multiple_returns_one_roll_per_term():
    rolls = DiceRoller.roll_multiple("2d6+1d8")
    assert [r.dice_type for r in rolls] == [6, 8]
    assert [len(r.results) for r in rolls] == [2, 1]


def test_roll_multiple_empty_notation():
    """Changed deliberately in PR-05: this returned [] before.

    Silently returning "no dice" for input the parser did not understand is the
    same failure mode as defaulting to 1d20 — it looks like a successful call.
    """
    with pytest.raises(ValueError):
        DiceRoller.roll_multiple("")


def test_roll_multiple_grand_total():
    rolls = DiceRoller.roll_multiple("3d6+2d10")
    grand_total = sum(r.total for r in rolls)
    assert 5 <= grand_total <= 38   # 3*1+2*1 .. 3*6+2*10


# --------------------------------------------------------------------------- #
# DiceRoll
# --------------------------------------------------------------------------- #

def test_dice_roll_str_format():
    roll = DiceRoll(dice_type=6, results=[3, 5], total=8)
    assert str(roll) == "2d6: [3, 5] = 8"
