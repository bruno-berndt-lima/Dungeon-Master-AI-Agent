# AI Dungeon Master

An interactive text-based D&D-style game powered by AI language models using LangGraph and LangChain.

## Overview

This project creates an AI-powered Dungeon Master that can run tabletop RPG-style adventures. The system uses LangGraph to orchestrate conversations between different AI agents, with the Dungeon Master agent managing the game world and story progression.

## Features

- AI Dungeon Master that manages game interactions and storytelling
- Turn-based gameplay with AI-controlled NPCs
- Dynamic narrative generation based on player actions
- Extensible agent-based architecture

## Technical Architecture

The project is built using:
- LangGraph for agent orchestration and conversation flow
- LangChain for interfacing with language models
- [Your chosen LLM] for powering the game's intelligence

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-dungeon-master.git
cd ai-dungeon-master

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Project Structure

- `src/agents/` - Contains all agent definitions including the Dungeon Master
- `src/graph/` - Defines the conversation graph and game state
- `src/models/` - LLM configuration and setup
- `src/prompts/` - System prompts for each agent
