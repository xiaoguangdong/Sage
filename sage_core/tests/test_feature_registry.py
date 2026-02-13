from __future__ import annotations

import pandas as pd

from sage_core.features import FEATURE_REGISTRY, PriceFeatures, MarketFeatures


def test_registry_has_defaults():
    registry = FEATURE_REGISTRY.list()
    assert "price_features" in registry
    assert "market_features" in registry


def test_price_features_transform():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=10, freq="D"),
        "stock": ["sh.600000"] * 10,
        "close": list(range(10, 20)),
    })
    features = PriceFeatures().transform(df)
    assert "mom_4w" in features.columns
    assert "relative_strength" in features.columns


def test_market_features_transform():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=30, freq="D"),
        "close": list(range(3000, 3030)),
        "high": list(range(3001, 3031)),
        "low": list(range(2999, 3029)),
        "amount": [1e8] * 30,
    })
    features = MarketFeatures().transform(df)
    assert "ret_4w" in features.columns
    assert "atr" in features.columns
