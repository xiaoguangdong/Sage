"""
日志规范化工具
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def _next_log_path(module_name: str, log_root: str = "logs") -> Path:
    date_str = datetime.now().strftime("%Y%m%d")
    module_dir = Path(log_root) / module_name
    module_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(module_dir.glob(f"{date_str}_*.log"))
    next_index = 1
    if existing:
        try:
            last = max(int(p.stem.split("_")[-1]) for p in existing)
            next_index = last + 1
        except ValueError:
            next_index = len(existing) + 1
    return module_dir / f"{date_str}_{next_index:03d}.log"


def setup_logging(module_name: str, level: int = logging.INFO, log_root: str = "logs") -> Path:
    log_path = _next_log_path(module_name, log_root)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = []
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)

    return log_path
