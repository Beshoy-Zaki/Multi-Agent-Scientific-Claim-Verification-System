"""Abstract base agent interface for MASCV agents."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class AgentConfig(dict):
    """Configuration dictionary that preserves full YAML structure while allowing flat access to the 'agent' section."""

    def get(self, key: str, default: Any = None) -> Any:
        if key in self:
            return super().get(key, default)
        agent = super().get("agent")
        if isinstance(agent, dict) and key in agent:
            return agent[key]
        return default

    def __getitem__(self, key: str) -> Any:
        if key in self:
            return super().__getitem__(key)
        agent = super().get("agent")
        if isinstance(agent, dict) and key in agent:
            return agent[key]
        return super().__getitem__(key)


class BaseAgent(ABC):
    """Abstract base class establishing standard lifecycle methods for all agents."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize base agent attributes and configuration."""
        self.name = name

        raw_config = config or {}
        self.config = AgentConfig(raw_config) if isinstance(raw_config, dict) else raw_config

    @property
    def agent_config(self) -> Dict[str, Any]:
        """Access the 'agent' section of the config, or the config itself if flat."""
        if isinstance(self.config, dict):
            agent = self.config.get("agent")
            if isinstance(agent, dict):
                return agent
        return self.config or {}

    @property
    def prompts(self) -> Dict[str, Any]:
        """Access prompt templates defined in configuration."""
        if isinstance(self.config, dict):
            p = self.config.get("prompts")
            if isinstance(p, dict):
                return p
        return {}

    @abstractmethod
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's core responsibility on the given state."""
        raise NotImplementedError
