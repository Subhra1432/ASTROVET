"""
Configuration management and YAML loading for ASTROVET.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    p = Path(config_path)
    if not p.is_file():
        raise FileNotFoundError(f"Configuration file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_yaml_config(config: Dict[str, Any], output_path: str | Path) -> Path:
    """Save configuration dictionary to a YAML file."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return p
