from __future__ import annotations

from pathlib import Path

from scripts.data._shared.runtime import get_tushare_root

TUSHARE_ROOT = Path(get_tushare_root())
MACRO_DIR = TUSHARE_ROOT / "macro"
NORTHBOUND_DIR = TUSHARE_ROOT / "northbound"
CONCEPTS_DIR = TUSHARE_ROOT / "concepts"
CONSTITUENTS_DIR = TUSHARE_ROOT / "constituents"
DAILY_DIR = TUSHARE_ROOT / "daily"
PRICE_STRUCTURE_DIR = TUSHARE_ROOT / "price_structure"
