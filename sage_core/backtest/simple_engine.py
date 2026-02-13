from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from .types import BacktestConfig, BacktestResult


@dataclass
class SimpleSignal:
    trade_date: str
    ts_code: str
    weight: float


class SimpleBacktestEngine:
    """
    简化回测引擎

    - 支持成本模型（单边 cost_rate）
    - 支持最大持仓数、行业上限
    - 采用等权持仓（或传入权重）
    - 仅用于研究回测骨架
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def run(self, signals: pd.DataFrame, returns: pd.DataFrame, industry_map: Optional[pd.DataFrame] = None) -> BacktestResult:
        """
        Args:
            signals: DataFrame columns = [trade_date, ts_code, score(optional), weight(optional), industry(optional)]
            returns: DataFrame columns = [trade_date, ts_code, ret]
            industry_map: DataFrame columns = [trade_date(optional), ts_code, industry]
        """
        required = {"trade_date", "ts_code"}
        if not required.issubset(signals.columns):
            raise ValueError(f"signals 缺少字段: {required - set(signals.columns)}")
        if not {"trade_date", "ts_code", "ret"}.issubset(returns.columns):
            raise ValueError("returns 需要字段: trade_date, ts_code, ret")

        signals = signals.copy()
        returns = returns.copy()
        returns["trade_date"] = returns["trade_date"].astype(str)
        signals["trade_date"] = signals["trade_date"].astype(str)

        if industry_map is not None and "industry" not in signals.columns:
            signals = signals.merge(industry_map, on=["ts_code"], how="left")

        if "weight" not in signals.columns:
            signals["weight"] = None

        portfolio_values = [self.config.initial_capital]
        portfolio_returns: List[float] = []
        trades: List[Dict] = []

        for trade_date, day_signals in signals.groupby("trade_date"):
            day_returns = returns[returns["trade_date"] == trade_date]
            if day_returns.empty:
                continue

            # 选股：按 score 排序或原顺序
            if "score" in day_signals.columns:
                day_signals = day_signals.sort_values("score", ascending=False)

            day_signals = day_signals.head(self.config.max_positions)

            # 行业上限控制
            if "industry" in day_signals.columns and self.config.max_industry_weight < 1.0:
                filtered = []
                industry_weight = {}
                for _, row in day_signals.iterrows():
                    industry = row.get("industry", "UNKNOWN")
                    industry_weight.setdefault(industry, 0.0)
                    if industry_weight[industry] + 1.0 > self.config.max_industry_weight * self.config.max_positions:
                        continue
                    industry_weight[industry] += 1.0
                    filtered.append(row)
                day_signals = pd.DataFrame(filtered)

            if day_signals.empty:
                portfolio_returns.append(0.0)
                portfolio_values.append(portfolio_values[-1])
                continue

            # 权重
            if day_signals["weight"].notna().any():
                weights = day_signals["weight"].fillna(0.0)
                weights = weights / weights.sum() if weights.sum() > 0 else pd.Series([1.0 / len(day_signals)] * len(day_signals))
            else:
                weights = pd.Series([1.0 / len(day_signals)] * len(day_signals))

            merged = day_signals.merge(day_returns, on=["trade_date", "ts_code"], how="left")
            merged["ret"] = merged["ret"].fillna(0.0)

            daily_ret = (merged["ret"] * weights.values).sum()

            # 成本（单边）
            daily_ret -= self.config.cost_rate

            portfolio_returns.append(daily_ret)
            portfolio_values.append(portfolio_values[-1] * (1 + daily_ret))

            trades.append({
                "trade_date": trade_date,
                "positions": len(merged),
                "return": daily_ret,
            })

        metrics = self._metrics(portfolio_returns, portfolio_values)
        return BacktestResult(
            returns=portfolio_returns,
            values=portfolio_values,
            trades=trades,
            metrics=metrics,
        )

    @staticmethod
    def _metrics(returns: List[float], values: List[float]) -> Dict[str, float]:
        if not returns:
            return {
                "total_return": 0.0,
                "annual_return": 0.0,
                "annual_volatility": 0.0,
                "max_drawdown": 0.0,
                "sharpe": 0.0,
            }
        total_return = values[-1] / values[0] - 1
        daily_mean = pd.Series(returns).mean()
        daily_std = pd.Series(returns).std()
        annual_return = (1 + daily_mean) ** 252 - 1
        annual_vol = daily_std * (252 ** 0.5)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0

        peak = values[0]
        max_dd = 0.0
        for v in values:
            peak = max(peak, v)
            dd = v / peak - 1
            max_dd = min(max_dd, dd)

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_vol,
            "max_drawdown": abs(max_dd),
            "sharpe": sharpe,
        }
