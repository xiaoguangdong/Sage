from __future__ import annotations

from pathlib import Path

import pandas as pd

from sage_core.data import DataCatalog, DataStore, DatasetSpec
from sage_core.utils import runtime_paths


def _setup_roots(tmp_path: Path):
    primary = tmp_path / "primary"
    secondary = tmp_path / "secondary"
    primary.mkdir(parents=True, exist_ok=True)
    secondary.mkdir(parents=True, exist_ok=True)
    return primary, secondary


def test_data_store_read_write(tmp_path, monkeypatch):
    primary, secondary = _setup_roots(tmp_path)
    monkeypatch.setenv("SAGE_DATA_ROOT_PRIMARY", str(primary))
    monkeypatch.setenv("SAGE_DATA_ROOT_SECONDARY", str(secondary))
    runtime_paths.reset_runtime_cache()

    spec = DatasetSpec(
        name="demo",
        section="raw",
        relative_path="tushare/demo.parquet",
        description="demo dataset",
    )
    store = DataStore(DataCatalog([spec]))

    df = pd.DataFrame({"value": [1, 2, 3]})
    path = store.write_parquet("demo", df)
    assert path.exists()

    loaded = store.read_parquet("demo")
    assert loaded["value"].sum() == 6
