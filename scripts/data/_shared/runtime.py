from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from sage_app.data.data_access import get_data_path as _data_access_get_data_path
from sage_app.data.data_access import get_data_root as _data_access_get_data_root
from sage_app.data.data_access import get_log_dir as _data_access_get_log_dir
from sage_app.data.data_access import get_tushare_root as _data_access_get_tushare_root
from sage_app.data.data_access import log_task_summary as _data_access_log_task_summary
from sage_app.data.data_access import next_log_path as _data_access_next_log_path
from sage_app.data.data_access import setup_task_logger

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ENV_LOADED = False
_BASE_CONFIG: Optional[Dict[str, Any]] = None


def add_project_root() -> Path:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    return PROJECT_ROOT


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

        with config_path.open("r", encoding="utf-8") as f:
            _BASE_CONFIG = yaml.safe_load(f) or {}
    except Exception:
        _BASE_CONFIG = {}
    return _BASE_CONFIG


def _resolve_root_path(value: Optional[str]) -> Optional[Path]:
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
    return _data_access_get_data_root(kind=kind)


def get_data_layout() -> Dict[str, str]:
    cfg = _load_base_config()
    layout_cfg = (cfg.get("data") or {}).get("layout") or {}
    return {
        "raw": layout_cfg.get("raw", "raw"),
        "processed": layout_cfg.get("processed", "processed"),
        "features": layout_cfg.get("features", "features"),
        "labels": layout_cfg.get("labels", "labels"),
        "signals": layout_cfg.get("signals", "signals"),
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
    return _data_access_get_data_path(section, *parts, root_kind=root_kind, ensure=ensure)


def get_tushare_root(root_kind: str = "primary", ensure: bool = False) -> Path:
    return _data_access_get_tushare_root(root_kind=root_kind, ensure=ensure)


def get_log_dir(module: str = "data") -> Path:
    return _data_access_get_log_dir(module=module)


def next_log_path(name: str, module: str = "data") -> Path:
    return _data_access_next_log_path(name=name, module=module)


def setup_logger(name: str, module: str = "data", level: int = logging.INFO) -> logging.Logger:
    return setup_task_logger(name=name, module=module, level=level)


def log_task_summary(
    logger: logging.Logger,
    task_name: str,
    window: Optional[str] = None,
    elapsed_s: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    _data_access_log_task_summary(logger, task_name, window=window, elapsed_s=elapsed_s, error=error)


def disable_proxy() -> None:
    keys = [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ]
    for key in keys:
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"


def get_tushare_token(explicit: Optional[str] = None) -> str:
    load_env_file()
    token = (explicit or os.getenv("TUSHARE_TOKEN") or os.getenv("TS_TOKEN") or "").strip()
    if token:
        return token
    raise RuntimeError("Missing Tushare token. Set env var TUSHARE_TOKEN (recommended) or TS_TOKEN.")


def load_env_file(path: Optional[Path] = None) -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    env_path = path or (PROJECT_ROOT / ".env")
    if env_path.exists():
        try:
            for raw in env_path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("'").strip('"')
                if key and key not in os.environ:
                    os.environ[key] = value
        except Exception:
            pass
    _ENV_LOADED = True
