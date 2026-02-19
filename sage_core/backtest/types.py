from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class BacktestConfig:
    """
    回测配置
    """

    initial_capital: float = 1_000_000
    cost_rate: float = 0.0005
    max_positions: int = 10
    max_industry_weight: float = 0.40
    t_plus_one: bool = True
    data_delay_days: int = 2


@dataclass
class BacktestResult:
    """
    回测结果
    """

    returns: List[float]
    values: List[float]
    trades: List[Dict]
    metrics: Dict[str, float]
    splits: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = field(default_factory=list)
    stop_loss_report: pd.DataFrame = field(default_factory=pd.DataFrame)
