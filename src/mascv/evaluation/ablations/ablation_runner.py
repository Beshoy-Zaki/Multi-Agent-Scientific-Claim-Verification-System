"""Runner for ablation experiment matrices."""

from typing import List, Dict, Any


class AblationRunner:
    """Executes ablation studies across various component configurations."""

    def run_all(self, benchmark_path: str) -> List[Dict[str, Any]]:
        """Run all configured ablation variations on benchmark data."""
        raise NotImplementedError("Ablation runner to be implemented.")
