from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """
    Load YAML/JSON config with a simple interface.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    if path.suffix in {".yaml", ".yml"}:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))

    raise ValueError(f"Unsupported config format: {path.suffix}")

