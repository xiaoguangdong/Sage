from __future__ import annotations

from pathlib import Path

import pandas as pd

from sage_core.data import DataCatalog
from scripts.data._shared import runtime as runtime_module


def _setup_roots(tmp_path: Path):
    primary = tmp_path / "primary"
    secondary = tmp_path / "secondary"
    primary.mkdir(parents=True, exist_ok=True)
    secondary.mkdir(parents=True, exist_ok=True)
    return primary, secondary


def test_catalog_resolve_secondary(tmp_path, monkeypatch):
    primary, secondary = _setup_roots(tmp_path)
    monkeypatch.setenv("SAGE_DATA_ROOT_PRIMARY", str(primary))
    monkeypatch.setenv("SAGE_DATA_ROOT_SECONDARY", str(secondary))
    runtime_module._BASE_CONFIG = None

    target = secondary / "raw" / "tushare" / "index" / "index_ohlc_all.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"a": [1]}).to_parquet(target, index=False)

    catalog = DataCatalog()
    resolved = catalog.resolve_path("index_ohlc_all")
    assert resolved == target


def test_catalog_resolve_primary_when_missing(tmp_path, monkeypatch):
    primary, secondary = _setup_roots(tmp_path)
    monkeypatch.setenv("SAGE_DATA_ROOT_PRIMARY", str(primary))
    monkeypatch.setenv("SAGE_DATA_ROOT_SECONDARY", str(secondary))
    runtime_module._BASE_CONFIG = None

    catalog = DataCatalog()
    resolved = catalog.resolve_path("index_ohlc_all")
    expected = primary / "raw" / "tushare" / "index" / "index_ohlc_all.parquet"
    assert resolved == expected
