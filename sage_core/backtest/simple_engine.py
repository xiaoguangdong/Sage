from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..portfolio.enhanced_risk_control import EnhancedRiskControl, RiskControlConfig
from .cost_model import TradingCostModel, create_default_cost_model
from .types import BacktestConfig, BacktestResult


@dataclass
class SimpleSignal:
    trade_date: str
    ts_code: str
    weight: float


class SimpleBacktestEngine:
    """
    简化回测引擎

    - 支持完整成本模型（固定成本 + 滑点 + 市场冲击）
    - 支持增强风险控制（动态仓位、ATR止损、行业止损、分档降仓）
    - 支持最大持仓数、行业上限
    - 采用等权持仓（或传入权重）
    - 仅用于研究回测骨架
    """

    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        cost_model: Optional[TradingCostModel] = None,
        use_advanced_cost: bool = True,
        use_enhanced_risk_control: bool = False,
        risk_control_config: Optional[RiskControlConfig] = None,
        entry_model=None,
    ):
        self.config = config or BacktestConfig()
        self.cost_model = cost_model or create_default_cost_model()
        self.use_advanced_cost = use_advanced_cost
        self.use_enhanced_risk_control = use_enhanced_risk_control
        self.risk_control = EnhancedRiskControl(risk_control_config) if use_enhanced_risk_control else None
        self.entry_model = entry_model  # EntryModelLR 实例（可选）

    def run(
        self, signals: pd.DataFrame, returns: pd.DataFrame, industry_map: Optional[pd.DataFrame] = None
    ) -> BacktestResult:
        """
        Args:
            signals: DataFrame columns = [trade_date, ts_code, score(optional), weight(optional), industry(optional), confidence(optional)]
            returns: DataFrame columns = [trade_date, ts_code, ret, high(optional), low(optional)]
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

        trading_dates = sorted(returns["trade_date"].unique())
        signals = self._shift_signals(signals, trading_dates, self.config.data_delay_days)

        signals_by_date = {date: day for date, day in signals.groupby("exec_date")} if not signals.empty else {}

        portfolio_values = [self.config.initial_capital]
        portfolio_returns: List[float] = []
        trades: List[Dict] = []

        holdings: Dict[str, float] = {}
        holdings_entry_prices: Dict[str, float] = {}  # 记录入场价格
        industry_returns: Dict[str, float] = {}  # 记录行业累计收益

        # EntryModel 历史数据缓存（最近60天，按股票分组）
        _history_window = 60
        _history_cache: Dict[str, List[Dict]] = {}

        for trade_date in trading_dates:
            day_returns = returns[returns["trade_date"] == trade_date]
            daily_ret = self._compute_portfolio_return(holdings, day_returns)
            turnover = 0.0

            # 更新 EntryModel 历史数据缓存
            if self.entry_model is not None and not day_returns.empty:
                for _, row in day_returns.iterrows():
                    ts_code = row["ts_code"]
                    record = {
                        "trade_date": trade_date,
                        "close": row.get("close", 0),
                        "turnover": row.get("turnover", 0),
                    }
                    if "high" in row.index:
                        record["high"] = row["high"]
                    if "low" in row.index:
                        record["low"] = row["low"]
                    _history_cache.setdefault(ts_code, []).append(record)
                    # 保持窗口大小
                    if len(_history_cache[ts_code]) > _history_window:
                        _history_cache[ts_code] = _history_cache[ts_code][-_history_window:]

            # 获取当日confidence（如果有）
            day_signals = signals_by_date.get(trade_date)
            confidence = self._get_confidence(day_signals)

            # 计算当前回撤
            if self.use_enhanced_risk_control and self.risk_control:
                self.risk_control.update_portfolio_value(portfolio_values[-1])
                current_drawdown = self.risk_control.get_current_drawdown()
            else:
                current_drawdown = None

            # 检查增强风控止损
            stop_stocks = []
            stop_industries = []
            if self.use_enhanced_risk_control and self.risk_control and holdings:
                # 检查个股ATR止损
                stop_stocks = self._check_stock_stop_loss(
                    holdings, holdings_entry_prices, day_returns, returns, trade_date
                )

                # 检查行业止损
                if "industry" in signals.columns:
                    stop_industries = self.risk_control.check_industry_stop_loss(industry_returns)

            # 调仓
            if not self.config.t_plus_one:
                holdings, turnover, holdings_entry_prices = self._rebalance(
                    holdings,
                    day_signals,
                    day_returns,
                    holdings_entry_prices,
                    confidence,
                    current_drawdown,
                    daily_ret,
                    stop_stocks,
                    stop_industries,
                    _history_cache if self.entry_model else None,
                )
                daily_ret = self._compute_portfolio_return(holdings, day_returns)
            else:
                holdings, turnover, holdings_entry_prices = self._rebalance(
                    holdings,
                    day_signals,
                    day_returns,
                    holdings_entry_prices,
                    confidence,
                    current_drawdown,
                    daily_ret,
                    stop_stocks,
                    stop_industries,
                    _history_cache if self.entry_model else None,
                )

            # 计算交易成本
            if self.use_advanced_cost:
                # 使用完整成本模型（固定成本 + 滑点 + 市场冲击）
                cost = self.cost_model.calculate_turnover_cost(
                    turnover=turnover,
                    portfolio_value=portfolio_values[-1],
                    market_data=day_returns,
                )
            else:
                # 使用简单成本率（向后兼容）
                cost = turnover * self.config.cost_rate

            daily_ret -= cost

            portfolio_returns.append(daily_ret)
            portfolio_values.append(portfolio_values[-1] * (1 + daily_ret))

            trades.append(
                {
                    "trade_date": trade_date,
                    "positions": len(holdings),
                    "turnover": turnover,
                    "cost": cost,
                    "return": daily_ret,
                    "confidence": confidence,
                    "stop_stocks": len(stop_stocks),
                    "stop_industries": len(stop_industries),
                }
            )

        metrics = self._metrics(portfolio_returns, portfolio_values)

        # 添加风控报告
        if self.use_enhanced_risk_control and self.risk_control:
            stop_loss_report = self.risk_control.get_stop_loss_report()
            metrics["stop_loss_events"] = len(stop_loss_report) if not stop_loss_report.empty else 0

        return BacktestResult(
            returns=portfolio_returns,
            values=portfolio_values,
            trades=trades,
            metrics=metrics,
        )

    def _shift_signals(self, signals: pd.DataFrame, trading_dates: List[str], delay_days: int) -> pd.DataFrame:
        if signals.empty:
            return signals.assign(exec_date=[])

        delay = max(0, int(delay_days))
        date_index = {d: i for i, d in enumerate(trading_dates)}
        exec_dates = []
        for date in signals["trade_date"]:
            idx = date_index.get(str(date))
            if idx is None:
                exec_dates.append(None)
                continue
            target_idx = idx + delay
            if target_idx >= len(trading_dates):
                exec_dates.append(None)
            else:
                exec_dates.append(trading_dates[target_idx])

        shifted = signals.copy()
        shifted["exec_date"] = exec_dates
        shifted = shifted[shifted["exec_date"].notna()]
        return shifted

    def _rebalance(
        self,
        holdings: Dict[str, float],
        day_signals: Optional[pd.DataFrame],
        day_returns: Optional[pd.DataFrame] = None,
        holdings_entry_prices: Optional[Dict[str, float]] = None,
        confidence: Optional[float] = None,
        current_drawdown: Optional[float] = None,
        daily_return: Optional[float] = None,
        stop_stocks: Optional[List[str]] = None,
        stop_industries: Optional[List[str]] = None,
        history_cache: Optional[Dict[str, List[Dict]]] = None,
    ) -> Tuple[Dict[str, float], float, Dict[str, float]]:
        if holdings_entry_prices is None:
            holdings_entry_prices = {}
        if stop_stocks is None:
            stop_stocks = []
        if stop_industries is None:
            stop_industries = []

        # 如果没有信号，清仓
        if day_signals is None or day_signals.empty:
            turnover = sum(abs(weight) for weight in holdings.values())
            return {}, turnover, {}

        day_signals = day_signals.copy()

        # 过滤止损股票和行业
        if stop_stocks:
            day_signals = day_signals[~day_signals["ts_code"].isin(stop_stocks)]
        if stop_industries and "industry" in day_signals.columns:
            day_signals = day_signals[~day_signals["industry"].isin(stop_industries)]

        # EntryModel 过滤（仅过滤新买入的股票，已持有的放行）
        if self.entry_model is not None and history_cache:
            candidate_codes = day_signals["ts_code"].tolist()
            stock_histories = {}
            for ts_code in candidate_codes:
                if ts_code in history_cache and len(history_cache[ts_code]) >= 30:
                    stock_histories[ts_code] = pd.DataFrame(history_cache[ts_code])
            if stock_histories:
                entry_signals = self.entry_model.predict_batch(stock_histories, existing_holdings=set(holdings.keys()))
                pass_codes = {code for code, sig in entry_signals.items() if sig}
                # 已持有的 + 历史数据不足的 也放行
                always_pass = set(holdings.keys()) | (set(candidate_codes) - set(stock_histories.keys()))
                keep_codes = pass_codes | always_pass
                day_signals = day_signals[day_signals["ts_code"].isin(keep_codes)]

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
            turnover = sum(abs(weight) for weight in holdings.values())
            return {}, turnover, {}

        # 权重
        if day_signals["weight"].notna().any():
            weights = day_signals["weight"].fillna(0.0).astype(float)
            weights = (
                weights / weights.sum() if weights.sum() > 0 else pd.Series([1.0 / len(day_signals)] * len(day_signals))
            )
        else:
            weights = pd.Series([1.0 / len(day_signals)] * len(day_signals))

        # 应用增强风控的仓位限制
        if self.use_enhanced_risk_control and self.risk_control:
            industries = day_signals["industry"].reset_index(drop=True) if "industry" in day_signals.columns else None
            weights = weights.reset_index(drop=True)
            weights = self.risk_control.apply_position_limits(weights, industries)

            # 计算动态仓位调整
            if confidence is not None:
                target_position = self.risk_control.compute_dynamic_position(
                    confidence=confidence,
                    current_drawdown=current_drawdown,
                    daily_return=daily_return,
                )
                # 整体缩放权重
                weights = weights * target_position

        new_holdings = {ts_code: float(weight) for ts_code, weight in zip(day_signals["ts_code"], weights)}

        # 更新入场价格
        new_entry_prices = holdings_entry_prices.copy()
        if day_returns is not None:
            for ts_code in new_holdings.keys():
                if ts_code not in holdings_entry_prices:
                    # 新买入，记录入场价
                    stock_ret = day_returns[day_returns["ts_code"] == ts_code]
                    if not stock_ret.empty and "close" in stock_ret.columns:
                        new_entry_prices[ts_code] = float(stock_ret["close"].iloc[0])

        # 清理已卖出股票的入场价
        for ts_code in list(new_entry_prices.keys()):
            if ts_code not in new_holdings:
                del new_entry_prices[ts_code]

        turnover = 0.0
        all_codes = set(holdings.keys()) | set(new_holdings.keys())
        for code in all_codes:
            turnover += abs(new_holdings.get(code, 0.0) - holdings.get(code, 0.0))

        return new_holdings, turnover, new_entry_prices

    @staticmethod
    def _compute_portfolio_return(holdings: Dict[str, float], day_returns: pd.DataFrame) -> float:
        if not holdings:
            return 0.0
        if day_returns is None or day_returns.empty:
            return 0.0
        weights_df = pd.DataFrame({"ts_code": list(holdings.keys()), "weight": list(holdings.values())})
        merged = weights_df.merge(day_returns, on="ts_code", how="left")
        merged["ret"] = merged["ret"].fillna(0.0)
        return float((merged["ret"] * merged["weight"]).sum())

    def _get_confidence(self, day_signals: Optional[pd.DataFrame]) -> float:
        """从信号中提取confidence，如果没有则返回默认值0.8"""
        if day_signals is None or day_signals.empty:
            return 0.8
        if "confidence" in day_signals.columns:
            conf = day_signals["confidence"].iloc[0]
            if pd.notna(conf):
                return float(conf)
        return 0.8

    def _check_stock_stop_loss(
        self,
        holdings: Dict[str, float],
        holdings_entry_prices: Dict[str, float],
        day_returns: pd.DataFrame,
        all_returns: pd.DataFrame,
        trade_date: str,
    ) -> List[str]:
        """检查个股ATR止损"""
        if not self.risk_control:
            return []

        # 构建持仓信息
        holdings_info = {}
        for ts_code, weight in holdings.items():
            entry_price = holdings_entry_prices.get(ts_code)
            if entry_price is not None:
                holdings_info[ts_code] = {
                    "entry_price": entry_price,
                    "weight": weight,
                }

        # 获取当日股票数据
        stock_data = day_returns.copy()
        if not stock_data.empty:
            stock_data = stock_data.set_index("ts_code")

        return self.risk_control.check_stock_stop_loss(stock_data, holdings_info)

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
        annual_vol = daily_std * (252**0.5)
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
