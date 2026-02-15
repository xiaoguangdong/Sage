from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BASE_CONFIG: Optional[Dict[str, Any]] = None


def get_project_root() -> Path:
    return PROJECT_ROOT


def _load_base_config() -> Dict[str, Any]:
    global _BASE_CONFIG
    if _BASE_CONFIG is not None:
        return _BASE_CONFIG

    config_path = PROJECT_ROOT / "config" / "base.yaml"
    if not config_path.exists():
        _BASE_CONFIG = {}
        return _BASE_CONFIG

    try:
        import yaml  # type: ignore

        with config_path.open("r", encoding="utf-8") as file:
            _BASE_CONFIG = yaml.safe_load(file) or {}
    except Exception:
        _BASE_CONFIG = {}
    return _BASE_CONFIG


def _resolve_path(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _dedupe_paths(paths: List[Optional[Path]]) -> List[Path]:
    deduped: List[Path] = []
    seen: set[str] = set()
    for path in paths:
        if path is None:
            continue
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _has_content(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    try:
        for child in path.iterdir():
            if child.is_file():
                return True
            if child.is_dir():
                for _ in child.iterdir():
                    return True
    except Exception:
        return False
    return False


def get_data_root(kind: str = "primary") -> Path:
    env_key = "SAGE_DATA_ROOT_PRIMARY" if kind == "primary" else "SAGE_DATA_ROOT_SECONDARY"
    env_value = os.getenv(env_key)

    cfg = _load_base_config()
    roots_cfg = (cfg.get("data") or {}).get("roots") or {}
    root_value = env_value or roots_cfg.get(kind)

    resolved = _resolve_path(root_value)
    if resolved and resolved.exists():
        return resolved

    fallback = PROJECT_ROOT / "data"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def get_data_layout() -> Dict[str, str]:
    cfg = _load_base_config()
    layout_cfg = (cfg.get("data") or {}).get("layout") or {}
    return {
        "raw": layout_cfg.get("raw", "raw"),
        "processed": layout_cfg.get("processed", "processed"),
        "features": layout_cfg.get("features", "features"),
        "labels": layout_cfg.get("labels", "labels"),
        "backtest": layout_cfg.get("backtest", "backtest"),
        "cache": layout_cfg.get("cache", "cache"),
        "states": layout_cfg.get("states", "states"),
        "meta": layout_cfg.get("meta", "meta"),
    }


def get_data_dir(*parts: str, root_kind: str = "primary", ensure: bool = False) -> Path:
    path = get_data_root(root_kind)
    for part in parts:
        path = path / part
    if ensure:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_path(section: str, *parts: str, root_kind: str = "primary", ensure: bool = False) -> Path:
    layout = get_data_layout()
    base = layout.get(section, section)
    return get_data_dir(base, *parts, root_kind=root_kind, ensure=ensure)


def _tushare_candidates(root_kind: str = "primary") -> List[Path]:
    cfg = _load_base_config()
    paths_cfg = (cfg.get("data") or {}).get("paths") or {}
    env_key = "SAGE_TUSHARE_ROOT" if root_kind == "primary" else "SAGE_TUSHARE_ROOT_SECONDARY"
    cfg_key = "tushare" if root_kind == "primary" else "tushare_secondary"

    candidates = _dedupe_paths([
        _resolve_path(os.getenv(env_key)),
        _resolve_path(paths_cfg.get(cfg_key)),
        get_data_root(root_kind) / "tushare",
        get_data_path("raw", "tushare", root_kind=root_kind),
    ])
    if candidates:
        return candidates
    return [get_data_root(root_kind) / "tushare"]


def get_tushare_root(root_kind: str = "primary", ensure: bool = False) -> Path:
    candidates = _tushare_candidates(root_kind=root_kind)
    if ensure:
        candidates[0].mkdir(parents=True, exist_ok=True)
        return candidates[0]
    canonical = candidates[0]
    legacy = candidates[-1]
    if canonical.exists():
        if legacy == canonical or not legacy.exists() or _has_content(canonical):
            return canonical
    if legacy.exists():
        return legacy
    for path in candidates[1:]:
        if path.exists():
            return path
    return canonical


def reset_runtime_cache() -> None:
    global _BASE_CONFIG
    _BASE_CONFIG = None
