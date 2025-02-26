from typing import Literal, Dict, Any, List, Optional, Tuple
from langchain_core.messages import SystemMessage
from src.utils.llm_logger import LLMLogger, LLMInteraction
from src.graph.game_state import GameState
from src.utils.dice import DiceRoller
from src.agents.base_agent import BaseAgent
from langgraph.types import Command
import re
from src.prompts.prompts import DICE_ROLLER_PROMPT
from src.models.llm import create_llm

class DiceRollRequest(Dict[str, Any]):
    """Structure for parsed dice roll requests."""
    dice_notation: str  # Standard dice notation like "2d20+1d8"
    modifier: Optional[int] = None  # Numerical modifier to add/subtract
    has_advantage: bool = False  # Roll with advantage (take higher of two rolls)
    has_disadvantage: bool = False  # Roll with disadvantage (take lower of two rolls)
    description: Optional[str] = None  # Description of what this roll is for

class DiceRollerAgent(BaseAgent):
    """Agent that handles rolling dice for game mechanics."""
    
    def __init__(self):
        super().__init__("dice_roller")
        self.system_prompt = DICE_ROLLER_PROMPT
        self.llm = create_llm()

    def get_definition(self) -> str:
        return self.system_prompt
    
    def process_task(self, state: GameState) -> Command[Literal["supervisor"]]:
        """Processes dice roll requests and returns results."""
        # Extract the dice roll request from the state
        latest_message = self._get_latest_message(state)
        
        # Parse the request directly
        dice_notation, modifier, has_advantage, has_disadvantage, description = self._parse_dice_request(latest_message)
        
        # Execute the dice roll using DiceRoller
        result_message = self._execute_dice_roll(
            dice_notation, 
            modifier, 
            has_advantage, 
            has_disadvantage, 
            description
        )
        
        # Update the state with the result
        updated_state = dict(state)
        if "messages" not in updated_state:
            updated_state["messages"] = []
            
        updated_state["messages"].append({
            "role": "assistant",
            "content": result_message,
            "name": self.agent_type
        })
        
        # Log the interaction
        self._log_interaction(
            query=latest_message,
            response=result_message,
            metadata={
                "dice_roll": {
                    "dice_notation": dice_notation,
                    "modifier": modifier,
                    "has_advantage": has_advantage,
                    "has_disadvantage": has_disadvantage,
                    "description": description
                }
            }
        )
        
        # Always return to supervisor after rolling dice
        return Command(goto="supervisor", update=updated_state)
    
    def _get_latest_message(self, state: GameState) -> str:
        """Extracts the latest message content from the state."""
        if "messages" in state and state["messages"]:
            latest = state["messages"][-1]
            if hasattr(latest, "content"):
                return latest.content
            return str(latest)
        elif "current_task" in state:
            return state["current_task"]
        return ""
    
    def _parse_dice_request(self, message: str) -> Tuple[str, int, bool, bool, str]:
        """Parse the dice request using regex patterns.
        
        Returns:
            Tuple with (dice_notation, modifier, has_advantage, has_disadvantage, description)
        """
        message = message.lower()
        
        # Extract dice notation patterns (2d6, d20, etc.)
        dice_notation = ""
        
        # Check for combined notation like "2d8+1d6"
        combined_pattern = r'(\d*d\d+(?:\s*\+\s*\d*d\d+)+)'
        combined_match = re.search(combined_pattern, message)
        
        if combined_match:
            # Found a combined notation like "2d8+1d6"
            dice_notation = combined_match.group(1).replace(" ", "")
        else:
            # Look for individual dice patterns and combine them
            dice_pattern = r'(\d*)d(\d+)'
            dice_matches = re.findall(dice_pattern, message)
            
            if dice_matches:
                dice_parts = []
                for count, sides in dice_matches:
                    count = count if count else "1"
                    dice_parts.append(f"{count}d{sides}")
                
                dice_notation = "+".join(dice_parts)
        
        # Default to d20 if no dice found
        if not dice_notation:
            dice_notation = "1d20"
        
        # Extract modifier (numbers to add/subtract)
        modifier = 0
        
        # Only look for modifiers that are NOT part of dice notation
        # This prevents 1d6 from "2d8 + 1d6" being interpreted as a +1 modifier
        clean_message = message
        
        # First remove all dice notation to avoid false positives
        for dice_part in re.findall(r'\d*d\d+', clean_message):
            clean_message = clean_message.replace(dice_part, "")
        
        # Now look for modifiers in the cleaned message
        modifier_pattern = r'(?:plus|add|with modifier|\+)\s*(\d+)'
        modifier_match = re.search(modifier_pattern, clean_message)
        
        if modifier_match:
            modifier = int(modifier_match.group(1))
        
        # Look for +N pattern that's not followed by "d"
        plus_pattern = r'\+\s*(\d+)(?!\s*d)'
        plus_match = re.search(plus_pattern, clean_message)
        if plus_match:
            modifier = int(plus_match.group(1))
        
        # Check for advantage/disadvantage keywords
        has_advantage = any(term in message for term in ["advantage", "adv", "take highest", "take higher"])
        has_disadvantage = any(term in message for term in ["disadvantage", "disadv", "take lowest", "take lower"])
        
        # Check for implicit advantage/disadvantage via "roll 2dX take highest/lowest" pattern
        # This handles cases like "roll 2d20 take the lowest"
        take_highest_pattern = r'(\d+)d(\d+).*(?:take|use|keep).*(?:high|higher|highest)'
        take_lowest_pattern = r'(\d+)d(\d+).*(?:take|use|keep).*(?:low|lower|lowest)'
        
        highest_match = re.search(take_highest_pattern, message)
        lowest_match = re.search(take_lowest_pattern, message)
        
        # If we find a "roll multiple and take highest/lowest" pattern, handle it
        if highest_match:
            count, sides = highest_match.groups()
            if count == "2":  # Only if rolling exactly 2 dice
                has_advantage = True
                # Adjust dice notation to use the base dice
                dice_notation = re.sub(fr'2d{sides}', f'1d{sides}', dice_notation)
        
        if lowest_match:
            count, sides = lowest_match.groups()
            if count == "2":  # Only if rolling exactly 2 dice
                has_disadvantage = True
                # Adjust dice notation to use the base dice
                dice_notation = re.sub(fr'2d{sides}', f'1d{sides}', dice_notation)
        
        # Extract description (what the roll is for)
        description = "dice roll"
        for marker in ["for", "to check", "to see if"]:
            if marker in message:
                parts = message.split(marker, 1)
                if len(parts) > 1:
                    # Clean up description - remove any stray quotes or brackets
                    desc = parts[1].strip().rstrip(".")
                    desc = re.sub(r'[\'\"{}]', '', desc)
                    description = desc
                    break
        
        print(f"Parsed request: dice={dice_notation}, mod={modifier}, adv={has_advantage}, disadv={has_disadvantage}, desc={description}")
        
        return dice_notation, modifier, has_advantage, has_disadvantage, description
    
    def _execute_dice_roll(self, dice_notation: str, modifier: int, 
                          has_advantage: bool, has_disadvantage: bool, 
                          description: str) -> str:
        """Execute the dice roll using the DiceRoller utility."""
        try:
            print(f"Processing dice roll: {dice_notation}, modifier: {modifier}, " +
                  f"advantage: {has_advantage}, disadvantage: {has_disadvantage}")
            
            # Handle advantage/disadvantage
            if has_advantage or has_disadvantage:
                # For advantage/disadvantage, determine the base dice type
                # Usually this is just d20, but we'll handle any dice type
                dice_match = re.search(r'(\d*)d(\d+)', dice_notation)
                if dice_match:
                    count = dice_match.group(1) or "1"
                    sides = dice_match.group(2)
                    base_roll = f"{count}d{sides}"
                else:
                    base_roll = dice_notation
                
                # Roll the dice twice
                print(f"Rolling twice for advantage/disadvantage: {base_roll}")
                first_result = DiceRoller.roll_multiple(base_roll)
                second_result = DiceRoller.roll_multiple(base_roll)
                
                # Calculate totals
                first_total = sum(roll.total for roll in first_result)
                second_total = sum(roll.total for roll in second_result)
                
                # Format details for display
                first_details = ", ".join(str(roll) for roll in first_result)
                second_details = ", ".join(str(roll) for roll in second_result)
                
                # Choose result based on advantage/disadvantage
                if has_advantage:
                    final_total = max(first_total, second_total)
                    advantage_text = f"with advantage (rolls: {first_total} and {second_total}, took higher)"
                else:  # disadvantage
                    final_total = min(first_total, second_total)
                    advantage_text = f"with disadvantage (rolls: {first_total} and {second_total}, took lower)"
                
                # Add modifier
                total_with_modifier = final_total + modifier
                modifier_text = f" + {modifier}" if modifier > 0 else f" - {abs(modifier)}" if modifier < 0 else ""
                
                # Format result
                if description:
                    result = f"🎲 Rolled {base_roll} {advantage_text}{modifier_text} for {description}: **{total_with_modifier}**"
                else:
                    result = f"🎲 Rolled {base_roll} {advantage_text}{modifier_text}: **{total_with_modifier}**"
                
                # Add roll details
                result += f"\nFirst roll: {first_details}\nSecond roll: {second_details}"
                
                return result
            
            # Standard dice rolls
            rolls = DiceRoller.roll_multiple(dice_notation)
            
            if not rolls:
                return "No valid dice roll found in the request."
                
            # Calculate total with modifier
            base_total = sum(roll.total for roll in rolls)
            total = base_total + modifier
            
            # Format result
            modifier_text = f" + {modifier}" if modifier > 0 else f" - {abs(modifier)}" if modifier < 0 else ""
            
            if description:
                result = f"🎲 Rolled {dice_notation}{modifier_text} for {description}: **{total}**"
            else:
                result = f"🎲 Rolled {dice_notation}{modifier_text}: **{total}**"
            
            # Add details about individual dice
            if len(rolls) == 1 and len(rolls[0].results) > 1:
                result += f" (rolled {rolls[0].results})"
            elif len(rolls) > 1:
                details = " + ".join(str(roll) for roll in rolls)
                result += f" ({details})"
            
            return result
                
        except Exception as e:
            print(f"Error in dice rolling: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error processing dice roll: {str(e)}"
