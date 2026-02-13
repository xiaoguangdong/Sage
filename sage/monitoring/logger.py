from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def _build_log_path(base_dir: Path, module: str) -> Path:
    """
    Build log file path as logs/<module>/<YYYYMMDD>_<NNN>.log.
    """
    module_dir = base_dir / module
    module_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    date_prefix = datetime.now().strftime("%Y%m%d")
    for idx in range(1, 1000):
        filename = f"{date_prefix}_{idx:03d}.log"
        path = module_dir / filename
        if not path.exists():
            return path
    raise RuntimeError("No available log file slot.")


def get_logger(
    module: str,
    name: Optional[str] = None,
    base_dir: str | Path = "logs",
) -> logging.Logger:
    """
    Create a logger writing to logs/<module>/<YYYYMMDD>_<NNN>.log.
    """
    base_dir = Path(base_dir)
    log_path = _build_log_path(base_dir, module)

    logger_name = name or f"sage.{module}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
