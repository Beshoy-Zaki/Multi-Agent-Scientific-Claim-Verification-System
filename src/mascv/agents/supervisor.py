"""Supervisor Agent: Manages overall investigation and controls workflow state."""

from typing import Any, Dict
from mascv.agents.base import BaseAgent


class SupervisorAgent(BaseAgent):
    """Orchestrates investigation iterations and decides whether additional cycles are needed."""

    def __init__(self, config: Dict[str, Any] = None) -> None:
        super().__init__(name="SupervisorAgent", config=config)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate claim states and dispatch next workflow action."""
        raise NotImplementedError("Supervisor decision logic to be implemented.")

    def decide_next_step(self, claim_state: Dict[str, Any]) -> str:
        """Determine if a claim requires more evidence or can proceed to final verdict."""
        raise NotImplementedError("Adaptive loop routing to be implemented.")
