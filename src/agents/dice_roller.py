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
    
    
    def _parse_dice_request(self, message: str) -> Tuple[str, int, bool, bool, str]:
        """Parse the dice request using LLM to extract structured information.
        
        Returns:
            Tuple with (dice_notation, modifier, has_advantage, has_disadvantage, description)
        """
        # Use a specialized prompt with examples to guide the LLM's understanding
        parse_prompt = """
        Extract the specific dice roll information from this request.

        EXAMPLES:
        For "Roll 1d20+5 for attack":
        {{
            "dice_notation": "1d20",
            "modifier": 5,
            "has_advantage": false,
            "has_disadvantage": false,
            "description": "attack"
        }}

        For "Roll 2d6 with advantage for stealth check":
        {{
            "dice_notation": "2d6",
            "modifier": 0,
            "has_advantage": true,
            "has_disadvantage": false,
            "description": "stealth check"
        }}

        For "Roll 1d20 + 2d6 for damage":
        {{
            "dice_notation": "1d20+2d6",
            "modifier": 0,
            "has_advantage": false,
            "has_disadvantage": false,
            "description": "damage"
        }}

        NOW PARSE THIS REQUEST: {message}

        ONLY respond with the JSON object, no explanation.
        """
        
        # Call the LLM for parsing
        parse_messages = [
            {"role": "system", "content": "You parse dice roll requests into exact dice notation, modifier values, and other attributes."},
            {"role": "user", "content": parse_prompt.format(message=message)}
        ]
        
        try:
            # Call the LLM
            response = self.llm.invoke(parse_messages)
            response_content = response.content if hasattr(response, "content") else str(response)
            
            print(f"Raw LLM response: {response_content}")
            
            # Clean up response - remove markdown code block markers and extract JSON
            response_content = re.sub(r'```(?:json)?\s*|\s*```', '', response_content)
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                response_content = json_match.group(0)
            
            # Parse the response
            import json
            parsed = json.loads(response_content)
            
            # Extract and validate dice notation
            dice_notation = parsed.get("dice_notation", "1d20")
            # Check if it still contains template placeholders
            if any(char in dice_notation for char in "PQXYpqxy"):
                # Looks like a template placeholder - extract from original message instead
                dice_pattern = r'(\d*d\d+(?:\s*\+\s*\d*d\d+)*)'
                dice_match = re.search(dice_pattern, message)
                dice_notation = dice_match.group(1).replace(" ", "") if dice_match else "1d20"
            
            # Handle modifier - convert string to int if needed
            modifier = parsed.get("modifier", 0)
            if isinstance(modifier, str):
                # Try to extract just the number from strings like "+5"
                mod_match = re.search(r'[-+]?\d+', modifier)
                modifier = int(mod_match.group(0)) if mod_match else 0
            else:
                modifier = int(modifier) if modifier is not None else 0
            
            # Handle advantage/disadvantage
            has_advantage = bool(parsed.get("has_advantage", False))
            has_disadvantage = bool(parsed.get("has_disadvantage", False))
            
            # Make sure we have a valid description
            description = parsed.get("description", "")
            if not description or description == "roll":
                # Try to extract purpose from the original message
                for marker in ["for", "to check", "to see if"]:
                    if marker in message:
                        parts = message.split(marker, 1)
                        if len(parts) > 1:
                            description = parts[1].strip()
                            break
                # If still no description, use default
                if not description or description == "roll":
                    description = "dice roll"
            
            print(f"Parsed request: dice={dice_notation}, mod={modifier}, adv={has_advantage}, disadv={has_disadvantage}, desc={description}")
            
            return dice_notation, modifier, has_advantage, has_disadvantage, description
            
        except Exception as e:
            print(f"Error parsing dice request with LLM: {str(e)}")
            # Safe defaults if parsing fails completely
            return "1d20", 0, False, False, "dice roll"
    
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
