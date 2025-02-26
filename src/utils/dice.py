import random
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
        """
        dice_parts = dice_str.lower().replace(" ", "").split("+")
        result = []
        
        for part in dice_parts:
            if "d" in part:
                quantity, dice_type = part.split("d")
                quantity = 1 if quantity == "" else int(quantity)
                result.append((quantity, int(dice_type)))
            
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