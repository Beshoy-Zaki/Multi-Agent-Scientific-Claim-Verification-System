"""Configuration loading and validation."""

from typing import Dict, Any
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and parse YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
