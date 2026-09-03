"""Tracks API token expenditures, financial cost, and execution latencies."""

from typing import Dict, Any


class CostLatencyTracker:
    """Monitors token usage, model inference costs, and wall-clock latencies."""

    def __init__(self) -> None:
        self.records: Dict[str, Any] = {}

    def log_call(self, agent_name: str, input_tokens: int, output_tokens: int, duration_sec: float) -> None:
        """Record usage data for a single agent turn."""
        raise NotImplementedError

    def summary(self) -> Dict[str, Any]:
        """Aggregate total cost and latency breakdown."""
        raise NotImplementedError
