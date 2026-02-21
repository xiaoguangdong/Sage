"""
日志规范化工具
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional


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


def format_task_summary(
    task_name: str,
    window: Optional[str] = None,
    elapsed_s: Optional[float] = None,
    error: Optional[str] = None,
) -> str:
    parts = [f"任务名={task_name}"]
    if window:
        parts.append(f"窗口={window}")
    if elapsed_s is not None:
        parts.append(f"耗时={elapsed_s:.1f}s")
    parts.append(f"失败原因={error or '无'}")
    return " | ".join(parts)
