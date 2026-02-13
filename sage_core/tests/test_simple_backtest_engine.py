from __future__ import annotations

import pandas as pd

from sage_core.backtest import BacktestConfig, SimpleBacktestEngine


def test_simple_engine_basic():
    signals = pd.DataFrame({
        "trade_date": ["20240102", "20240102", "20240103", "20240103"],
        "ts_code": ["000001.SZ", "000002.SZ", "000001.SZ", "000002.SZ"],
        "score": [0.9, 0.8, 0.95, 0.7],
    })

    returns = pd.DataFrame({
        "trade_date": ["20240102", "20240102", "20240103", "20240103"],
        "ts_code": ["000001.SZ", "000002.SZ", "000001.SZ", "000002.SZ"],
        "ret": [0.01, -0.005, 0.02, 0.01],
    })

    config = BacktestConfig(initial_capital=1000, cost_rate=0.0, max_positions=2)
    engine = SimpleBacktestEngine(config)
    result = engine.run(signals, returns)

    assert len(result.returns) == 2
    assert result.values[-1] > 1000
