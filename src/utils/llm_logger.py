import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict

@dataclass
class LLMInteraction:
    """Represents a single interaction with the LLM"""
    timestamp: str
    agent: str
    query: str
    response: str
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def create(cls, agent: str, query: str, response: str, metadata: Optional[Dict[str, Any]] = None):
        return cls(
            timestamp=datetime.now().isoformat(),
            agent=agent,
            query=query,
            response=response,
            metadata=metadata or {}
        )

class LLMLogger:
    """Handles logging of LLM interactions"""
    
    def __init__(self, log_dir: str = "logs/llm_interactions"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_log_file = self._get_current_log_file()
    
    def _get_current_log_file(self) -> Path:
        """Get the current day's log file"""
        current_date = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"llm_log_{current_date}.jsonl"
    
    def log_interaction(self, interaction: LLMInteraction):
        """Log a single interaction"""
        # Update log file if day has changed
        self.current_log_file = self._get_current_log_file()
        
        # Write interaction to log file
        with self.current_log_file.open("a", encoding="utf-8") as f:
            json.dump(asdict(interaction), f)
            f.write("\n")
    
    def get_recent_interactions(self, limit: int = 100) -> list[LLMInteraction]:
        """Retrieve recent interactions"""
        interactions = []
        
        # Read from current day's log file
        if self.current_log_file.exists():
            with self.current_log_file.open("r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    interactions.append(LLMInteraction(**data))
        
        return interactions[-limit:] 