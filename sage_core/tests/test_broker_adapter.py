from __future__ import annotations

import pandas as pd
import pytest

from sage_core.execution.broker_adapter import (
    build_orders_from_portfolio,
    create_broker_adapter,
    normalize_ts_code,
)


def test_normalize_ts_code():
    assert normalize_ts_code("600519") == "600519.SH"
    assert normalize_ts_code("000001") == "000001.SZ"
    assert normalize_ts_code("430047") == "430047.BJ"
    assert normalize_ts_code("000001.SZ") == "000001.SZ"


def test_build_orders_from_portfolio():
    portfolio = pd.DataFrame(
        [
            {"code": "600519", "weight": 0.3},
            {"code": "000001", "weight": 0.2},
            {"code": "000002", "weight": 0.0},
        ]
    )
    orders = build_orders_from_portfolio(portfolio, top_n=10)
    assert len(orders) == 2
    assert orders[0].ts_code == "600519.SH"
    assert orders[0].target_weight == 0.3


def test_pingan_adapter_dry_run_only():
    adapter = create_broker_adapter("pingan", config={})
    orders = build_orders_from_portfolio(pd.DataFrame([{"ts_code": "000001.SZ", "target_weight": 0.2}]))
    result = adapter.submit_orders(orders, dry_run=True)
    assert result.dry_run is True
    assert result.accepted_orders == 1

    with pytest.raises(NotImplementedError):
        adapter.submit_orders(orders, dry_run=False)
