from __future__ import annotations

import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def add_project_root() -> Path:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    return PROJECT_ROOT


def get_project_root() -> Path:
    return PROJECT_ROOT


def get_data_dir(*parts: str) -> Path:
    path = PROJECT_ROOT / "data"
    for part in parts:
        path = path / part
    return path


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


def get_tushare_token(explicit: Optional[str] = None) -> str:
    token = (explicit or os.getenv("TUSHARE_TOKEN") or os.getenv("TS_TOKEN") or "").strip()
    if token:
        return token
    raise RuntimeError(
        "Missing Tushare token. Set env var TUSHARE_TOKEN (recommended) or TS_TOKEN."
    )
