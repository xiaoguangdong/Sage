from __future__ import annotations

from pathlib import Path

import pandas as pd

from sage_core.data import DataCatalog, DataStore, DatasetSpec
from sage_core.features import FeaturePipeline, PriceFeatures
from sage_core.utils import runtime_paths


def _setup_roots(tmp_path: Path):
    primary = tmp_path / "primary"
    secondary = tmp_path / "secondary"
    primary.mkdir(parents=True, exist_ok=True)
    secondary.mkdir(parents=True, exist_ok=True)
    return primary, secondary


def test_feature_pipeline_run(tmp_path, monkeypatch):
    primary, secondary = _setup_roots(tmp_path)
    monkeypatch.setenv("SAGE_DATA_ROOT_PRIMARY", str(primary))
    monkeypatch.setenv("SAGE_DATA_ROOT_SECONDARY", str(secondary))
    runtime_paths.reset_runtime_cache()

    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=10, freq="D"),
        "stock": ["sh.600000"] * 10,
        "close": list(range(10, 20)),
    })

    spec = DatasetSpec(
        name="feature_demo",
        section="features",
        relative_path="demo/feature_demo.parquet",
        description="feature pipeline demo",
    )
    store = DataStore(DataCatalog([spec]))

    pipeline = FeaturePipeline([PriceFeatures()], name="price_pipeline")
    result = pipeline.run(df, store=store, dataset_name="feature_demo")

    assert "mom_4w" in result.feature_columns
    assert store.resolve("feature_demo").exists()
