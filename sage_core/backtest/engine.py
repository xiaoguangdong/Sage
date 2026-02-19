from __future__ import annotations

from typing import Protocol

import pandas as pd

from .types import BacktestConfig, BacktestResult


class BacktestEngine(Protocol):
    """
    回测引擎协议
    """

    def run(self, df: pd.DataFrame, config: BacktestConfig) -> BacktestResult: ...
