"""Abstract base agent interface for MASCV agents."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseAgent(ABC):
    """Abstract base class establishing standard lifecycle methods for all agents."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize base agent attributes and configuration."""
        self.name = name

        raw_config = config or {}

        # The YAML files under config/agents/*.yaml are shaped as
        # {"agent": {"model": ..., "temperature": ..., "parameters": {...}}}.
        # Unwrap that nesting here so `self.config.get("model")` works
        # whether the caller passes a flat dict (as tests do) or a loaded
        # agent YAML config directly.
        if isinstance(raw_config, dict) and "agent" in raw_config:
            self.config = raw_config["agent"]
        else:
            self.config = raw_config

    @abstractmethod
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's core responsibility on the given state."""
        raise NotImplementedError
