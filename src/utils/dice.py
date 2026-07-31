import random
import re
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class DiceRoll:
    """Represents a single dice roll result"""
    dice_type: int  # The type of dice (6, 8, 20, etc.)
    results: List[int]  # Individual roll results
    total: int  # Sum of all rolls

    def __str__(self):
        return f"{len(self.results)}d{self.dice_type}: {self.results} = {self.total}"

class DiceRoller:
    """Handles all dice rolling operations"""
    
    @staticmethod
    def parse_dice_string(dice_str: str) -> List[Tuple[int, int]]:
        """
        Parses a dice notation string into a list of (quantity, dice_type) tuples
        Example: "2d6 + 1d8" -> [(2, 6), (1, 8)]

        Flat numeric terms are modifiers, not dice, and are skipped —
        ``"2d6+3"`` and ``"2d6-1"`` both yield ``[(2, 6)]``. The caller adds the
        modifier; ``DiceRollerAgent`` extracts it separately.

        Raises:
            ValueError: on notation this return type cannot represent — a
                subtracted dice term (``"2d6-1d4"``), a non-positive quantity or
                die size, or anything unparseable. Returning something plausible
                for input we do not understand is how a roll silently becomes
                the wrong roll.
        """
        text = dice_str.lower().replace(" ", "")
        if not text:
            raise ValueError("empty dice notation")

        # Split while keeping each term's sign. "2d6-1d4" must be rejected
        # rather than quietly rolled as "2d6+1d4".
        terms = re.findall(r"[+-]?[^+-]+", text)
        result = []

        for term in terms:
            negative = term.startswith("-")
            body = term.lstrip("+-")
            if not body:
                raise ValueError(f"malformed dice notation: {dice_str!r}")

            if "d" not in body:
                if not body.isdigit():
                    raise ValueError(f"malformed dice notation: {dice_str!r}")
                continue  # a flat modifier — the caller owns it

            if negative:
                raise ValueError(
                    f"cannot subtract dice: {term!r} in {dice_str!r}"
                )

            quantity_text, _, sides_text = body.partition("d")
            quantity = 1 if quantity_text == "" else int(quantity_text)
            sides = int(sides_text)

            if quantity < 1 or sides < 1:
                raise ValueError(f"cannot roll {quantity}d{sides}")

            result.append((quantity, sides))

        if not result:
            raise ValueError(f"no dice in notation: {dice_str!r}")

        return result

    @staticmethod
    def roll_single_type(quantity: int, dice_type: int) -> DiceRoll:
        """Rolls a specific quantity of a single dice type"""
        results = [random.randint(1, dice_type) for _ in range(quantity)]
        return DiceRoll(
            dice_type=dice_type,
            results=results,
            total=sum(results)
        )

    @staticmethod
    def roll_multiple(dice_str: str) -> List[DiceRoll]:
        """
        Rolls multiple dice of different types
        Example: "2d6 + 1d8" -> [DiceRoll(2d6), DiceRoll(1d8)]
        """
        dice_combinations = DiceRoller.parse_dice_string(dice_str)
        return [
            DiceRoller.roll_single_type(quantity, dice_type)
            for quantity, dice_type in dice_combinations
        ] 