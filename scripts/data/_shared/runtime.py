from __future__ import annotations

import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

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


def get_data_root(kind: str = "primary") -> Path:
    env_key = "SAGE_DATA_ROOT_PRIMARY" if kind == "primary" else "SAGE_DATA_ROOT_SECONDARY"
    env_value = os.getenv(env_key)
    cfg = _load_base_config()
    roots_cfg = (cfg.get("data") or {}).get("roots") or {}
    root_value = env_value or roots_cfg.get(kind)

    resolved = _resolve_root_path(root_value)
    if resolved and resolved.exists():
        return resolved

    # fallback to repo data dir
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


def get_tushare_root(root_kind: str = "primary", ensure: bool = False) -> Path:
    return get_data_path("raw", "tushare", root_kind=root_kind, ensure=ensure)


def get_log_dir(module: str = "data") -> Path:
    log_dir = PROJECT_ROOT / "logs" / module
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def next_log_path(name: str, module: str = "data") -> Path:
    log_dir = get_log_dir(module)
    date_str = datetime.now().strftime("%Y%m%d")
    pattern = re.compile(rf"^{date_str}_(\d{{3}})_{re.escape(name)}\.log$")
    next_seq = 1
    for path in log_dir.iterdir():
        match = pattern.match(path.name)
        if match:
            next_seq = max(next_seq, int(match.group(1)) + 1)
    return log_dir / f"{date_str}_{next_seq:03d}_{name}.log"


def setup_logger(name: str, module: str = "data", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger

    log_path = next_log_path(name, module)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


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
    raise RuntimeError(
        "Missing Tushare token. Set env var TUSHARE_TOKEN (recommended) or TS_TOKEN."
    )


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
