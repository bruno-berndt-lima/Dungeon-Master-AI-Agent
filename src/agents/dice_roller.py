from typing import Literal, Tuple
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from src.utils.llm_logger import LLMLogger, LLMInteraction
from src.graph.game_state import GameState
from src.utils.dice import DiceRoller
from src.agents.base_agent import BaseAgent
from langgraph.graph import END
from langgraph.types import Command
import re
from src.prompts.prompts import DICE_PARSE_PROMPT, DICE_ROLLER_PROMPT
from src.models.llm import create_llm

class DiceRequest(TypedDict):
    """The fields a dice request parses into.

    Replaces `DiceRollRequest(Dict[str, Any])`, which declared class attributes
    with defaults on a `Dict` subclass — not a usable structure, and never
    instantiated (KNOWN_ISSUES #15). This one is the actual output schema.
    """
    dice_notation: str
    modifier: int
    has_advantage: bool
    has_disadvantage: bool
    description: str


class DiceParseError(ValueError):
    """The request could not be read as a dice roll."""


# A dice expression is a formal language, so read it with a parser rather than a
# model. Measured, `llama3.2:3b` invents modifiers — `+1` on "2d8 + 1d6", `+5` on
# a plain advantage roll, `+2` on "roll for initiative" — and every one of those
# silently changes the number the player gets. `qwen2.5:7b` gets them right but
# costs 9.5 s against 4.6 s. Neither trade is necessary: the request already says
# exactly what to roll.
DICE_EXPRESSION = re.compile(
    r"\d*d\d+(?:\s*[+-]\s*(?:\d*d\d+|\d+))*", re.IGNORECASE
)
FLAT_MODIFIER = re.compile(r"([+-])\s*(\d+)(?!\s*d\d)", re.IGNORECASE)
# "disadvantage" contains "advantage", but the word boundary keeps them apart.
ADVANTAGE = re.compile(r"\badvantage\b", re.IGNORECASE)
DISADVANTAGE = re.compile(r"\bdisadvantage\b", re.IGNORECASE)
PURPOSE = re.compile(r"\b(?:for|to check|to see if)\b\s+(.+)$", re.IGNORECASE)


def extract_dice_expression(message: str):
    """Pull `(notation, modifier)` out of a request, or `(None, 0)`.

    Only reads what is literally written. A request with no flat number has no
    modifier — there is nothing for a model to infer.
    """
    match = DICE_EXPRESSION.search(message or "")
    if not match:
        return None, 0

    expression = match.group(0).replace(" ", "")

    dice_terms = []
    for sign, term in re.findall(r"([+-]?)(\d*d\d+)", expression, re.IGNORECASE):
        if sign == "-":
            raise DiceParseError(
                f"cannot subtract dice: {sign}{term} in {expression!r}"
            )
        # Normalise "d20" to "1d20" so the result line reads the same either way.
        dice_terms.append(term if term[0].lower() != "d" else f"1{term}")

    if not dice_terms:
        return None, 0

    modifier = sum(
        int(f"{sign}{value}") for sign, value in FLAT_MODIFIER.findall(expression)
    )
    return "+".join(dice_terms), modifier


def extract_roll_flags(message: str):
    """Advantage, disadvantage, and stated purpose — all keyword-exact."""
    text = message or ""
    disadvantage = bool(DISADVANTAGE.search(text))
    advantage = bool(ADVANTAGE.search(text)) and not disadvantage

    purpose = PURPOSE.search(text)
    description = purpose.group(1).strip().rstrip(".") if purpose else ""
    if description.lower().startswith(("advantage", "disadvantage")):
        description = ""

    return advantage, disadvantage, description


class DiceRollerAgent(BaseAgent):
    """Agent that handles rolling dice for game mechanics."""

    def __init__(self):
        super().__init__("dice_roller")
        self.system_prompt = DICE_ROLLER_PROMPT
        self.llm = create_llm(self.agent_type)
        self.parser = self.llm.with_structured_output(
            DiceRequest, method="json_schema"
        )

    def get_definition(self) -> str:
        return self.system_prompt
    
    def process_task(self, state: GameState) -> Command[Literal["__end__"]]:
        """Processes dice roll requests and returns results."""
        # Extract the dice roll request from the state
        latest_message = self._get_latest_message(state)
        
        try:
            dice_notation, modifier, has_advantage, has_disadvantage, description = (
                self._parse_dice_request(latest_message)
            )
        except DiceParseError as exc:
            # Say what went wrong. The old code fell back to 1d20 here, which
            # answered an unasked question with a confident number.
            result_message = (
                f"I could not tell what to roll. Try dice notation — "
                f"`2d6+3`, `1d20 with advantage`. ({exc})"
            )
            self._log_interaction(
                query=latest_message,
                response=result_message,
                metadata={"error": str(exc)},
            )
            return Command(
                goto=END,
                update={
                    "messages": [AIMessage(content=result_message, name=self.agent_type)],
                    "last_response": result_message,
                },
            )

        # Execute the dice roll using DiceRoller
        result_message = self._execute_dice_roll(
            dice_notation,
            modifier,
            has_advantage,
            has_disadvantage,
            description
        )

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

        # Terminate rather than returning to the supervisor. A roll is a complete
        # answer, and handing back meant the supervisor re-routed on the roll
        # *result* and sent it to `researcher` — roughly 40 s of generation, at
        # 5.3 tok/s, answering a question nobody asked (KNOWN_ISSUES #6).
        #
        # Return only the message this node produced (PR-03 — this used to
        # `dict(state)`, a shallow copy, then append to the caller's own list).
        return Command(
            goto=END,
            update={
                "messages": [AIMessage(content=result_message, name=self.agent_type)],
                "last_response": result_message,
            },
        )
    
    
    def _parse_dice_request(self, message: str) -> Tuple[str, int, bool, bool, str]:
        """Parse the dice request into structured fields.

        This used to be ~70 lines of defence against the model: stripping
        markdown fences, regex-extracting the first ``{...}``, detecting
        ``PQXYpqxy`` template placeholders the model had copied from the
        few-shot examples instead of substituting, coercing a string ``"+5"``
        into an int, and salvaging a description by splitting the message on
        ``for`` / ``to check``. Every line of it existed because a small local
        model could not be trusted to emit clean JSON unaided. Schema-constrained
        decoding removes the need — the fields come back typed, or not at all.

        The model is now the fallback, not the first step. Anything the request
        states literally — the dice, the modifier, advantage, the purpose — is
        read directly. The model is asked only when the request names no dice at
        all ("roll for initiative"), and even then the modifier and the
        advantage flags stay deterministic, because a number the request does not
        contain is a number the model invented.

        Returns:
            Tuple with (dice_notation, modifier, has_advantage, has_disadvantage, description)
        """
        try:
            notation, modifier = extract_dice_expression(message)
            has_advantage, has_disadvantage, description = extract_roll_flags(message)

            if notation is None:
                parsed = self.parser.invoke([
                    SystemMessage(content=DICE_PARSE_PROMPT),
                    HumanMessage(content=message),
                ])
                if not isinstance(parsed, dict):
                    raise ValueError(f"parser returned {parsed!r}")

                notation = str(parsed.get("dice_notation") or "").strip()
                if not notation:
                    raise ValueError("no dice notation in the parsed request")

                # The model is allowed to name the dice, and nothing else. It
                # will fold a bonus into the notation given the chance —
                # "roll for initiative" came back as "1d20+2" — so the notation
                # is re-read and the modifier is taken from the player's own
                # words, which is the only place a real one can come from.
                notation, _ = extract_dice_expression(notation)
                if notation is None:
                    raise ValueError("the model returned no rollable dice")
                modifier = sum(
                    int(f"{sign}{value}")
                    for sign, value in FLAT_MODIFIER.findall(message.replace(" ", ""))
                )

                description = description or str(parsed.get("description") or "").strip()

            # The notation has to be rollable. Typed does not mean meaningful.
            DiceRoller.parse_dice_string(notation)

            return notation, modifier, has_advantage, has_disadvantage, description
        except Exception as exc:
            # A dice request that cannot be parsed must not silently become
            # 1d20 — that returns a confident number for a roll nobody asked
            # for. Let the caller report it.
            raise DiceParseError(
                f"could not read a dice roll from {message!r}: {exc}"
            ) from exc
    
    def _execute_dice_roll(self, dice_notation: str, modifier: int, 
                          has_advantage: bool, has_disadvantage: bool, 
                          description: str) -> str:
        """Execute the dice roll using the DiceRoller utility."""
        try:
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
            return f"Error processing dice roll: {str(e)}"
