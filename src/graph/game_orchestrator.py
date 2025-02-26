from typing import Annotated, Sequence, TypedDict, Union, Dict, List, Literal
from langgraph.graph import Graph, StateGraph
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import MessagesState, START, END
from langgraph.types import Command
from src.actors.player import Player
from src.actors.npc import NPC
from src.graph.game_state import GameState
from src.agents.base_agent import BaseAgent
from src.agents.dungeon_master import DungeonMaster
from src.agents.researcher import ResearcherAgent
from src.agents.dice_roller import DiceRollerAgent
from src.agents.supervisor import GameSupervisor
from src.models.llm import create_llm

def create_game_graph():
    """Creates the main game orchestration graph using agent nodes."""
    # Initialize all agents
    supervisor = GameSupervisor()
    dungeon_master = DungeonMaster()
    researcher = ResearcherAgent()
    dice_roller = DiceRollerAgent()

    # Create workflow
    workflow = StateGraph(GameState)
    
    # Add nodes - directly use the process_task methods
    workflow.add_node("supervisor", supervisor.process_task)
    workflow.add_node("dungeon_master", dungeon_master.process_task)
    workflow.add_node("researcher", researcher.process_task)
    workflow.add_node("dice_roller", dice_roller.process_task)
    
    # Set the entrypoint
    workflow.set_entry_point("supervisor")
    
    return workflow.compile()