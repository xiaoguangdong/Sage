from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from sage_core.utils.logging_utils import format_task_summary
from sage_core.utils.runtime_paths import get_data_path as _get_data_path
from sage_core.utils.runtime_paths import get_data_root as _get_data_root
from sage_core.utils.runtime_paths import get_tushare_root as _get_tushare_root

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_data_root(kind: str = "primary") -> Path:
    return _get_data_root(kind=kind)


def get_data_path(section: str, *parts: str, root_kind: str = "primary", ensure: bool = False) -> Path:
    return _get_data_path(section, *parts, root_kind=root_kind, ensure=ensure)


def get_tushare_root(root_kind: str = "primary", ensure: bool = False) -> Path:
    return _get_tushare_root(root_kind=root_kind, ensure=ensure)


def get_log_dir(module: str = "data") -> Path:
    log_dir = PROJECT_ROOT / "logs" / module
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def next_log_path(name: str, module: str = "data") -> Path:
    log_dir = get_log_dir(module)
    date_str = datetime.now().strftime("%Y%m%d")
    pattern = f"{date_str}_"
    candidates = [p for p in log_dir.iterdir() if p.is_file() and p.name.startswith(pattern)]
    next_seq = 1
    if candidates:
        for path in candidates:
            stem = path.stem
            parts = stem.split("_")
            if len(parts) >= 2 and parts[0] == date_str:
                try:
                    next_seq = max(next_seq, int(parts[1]) + 1)
                except ValueError:
                    continue
    return log_dir / f"{date_str}_{next_seq:03d}_{name}.log"


def setup_task_logger(name: str, module: str = "data", level: int = logging.INFO) -> logging.Logger:
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


def log_task_summary(
    logger: logging.Logger,
    task_name: str,
    window: Optional[str] = None,
    elapsed_s: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    logger.info(format_task_summary(task_name, window=window, elapsed_s=elapsed_s, error=error))
