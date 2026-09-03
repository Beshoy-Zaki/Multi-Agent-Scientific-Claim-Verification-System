"""Abstract base agent interface for MASCV agents."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseAgent(ABC):
    """Abstract base class establishing standard lifecycle methods for all agents."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize base agent attributes and configuration."""
        self.name = name
        self.config = config or {}

    @abstractmethod
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's core responsibility on the given state."""
        raise NotImplementedError
