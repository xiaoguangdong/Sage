from __future__ import annotations

from datetime import datetime


def today_str() -> str:
    """Return today's date in YYYYMMDD format."""
    return datetime.now().strftime("%Y%m%d")

