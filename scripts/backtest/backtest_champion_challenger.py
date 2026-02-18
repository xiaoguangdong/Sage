#!/usr/bin/env python3
"""
Champion/Challenger 回测验证

功能：
1. 对 Champion（豆包四因子）和 4 个 Challenger 策略分别进行 Walk-forward 回测
2. 计算各策略的表现指标（收益、夏普、最大回撤、IC等）
3. 生成对比评估报告
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.data._shared.runtime import get_data_root, get_tushare_root
from sage_core.stock_selection.stock_selector import StockSelector, SelectionConfig
from sage_core.stock_selection.multi_alpha_selector import MultiAlphaStockSelector, MultiAlphaConfig
from sage_core.stock_selection.regime_stock_selector import (
    RegimeSelectionConfig, RegimeStockSelector, REGIME_NAMES,
)
from sage_core.governance.strategy_governance import (
    ChallengerConfig,
    MultiAlphaChallengerStrategies,
    SeedBalanceStrategy,
    StrategyGovernanceConfig,
    normalize_strategy_id,
    SIGNAL_SCHEMA,
)
from sage_core.backtest.types import BacktestResult


# ==================== 数据加载工具 ====================

def _resolve_tushare_root(data_dir: Optional[str]) -> Path:
    if data_dir:
        return Path(data_dir)
    return get_tushare_root()


def _find_trade_dates(daily_dir: Path, start_date: str, end_date: str) -> List[str]:
    """获取指定日期范围内的交易日列表"""
    dates = set()
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    
    for year in range(start_year - 1, end_year + 2):
        path = daily_dir / f"daily_{year}.parquet"
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path, columns=["trade_date"])
            dates.update(df["trade_date"].astype(str).unique())
        except Exception:
            continue
    
    dates = sorted([d for d in dates if start_date <= d <= end_date])
    return dates


def _load_daily_data(daily_dir: Path, trade_dates: List[str]) -> pd.DataFrame:
    """加载日线数据"""
    frames = []
    years = set(int(d[:4]) for d in trade_dates)
    
    for year in years:
        path = daily_dir / f"daily_{year}.parquet"
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
            frames.append(df)
        except Exception:
            continue
    
    if not frames:
        return pd.DataFrame()
    
    df = pd.concat(frames, ignore_index=True)
    df = df[df["trade_date"].astype(str).isin(trade_dates)]
    return df


def _load_daily_basic(daily_basic_path: Path, trade_dates: List[str]) -> pd.DataFrame:
    """加载每日基本面数据"""
    if not daily_basic_path.exists():
        return pd.DataFrame()
    
    try:
        df = pd.read_parquet(daily_basic_path)
        df = df[df["trade_date"].astype(str).isin(trade_dates)]
        return df
    except Exception:
        return pd.DataFrame()


def _load_fina_indicator(fina_dir: Path, end_date: str) -> pd.DataFrame:
    """加载财务指标数据"""
    frames = []
    year = int(end_date[:4])
    
    for y in [year, year - 1, year - 2]:
        path = fina_dir / f"fina_indicator_{y}.parquet"
        if path.exists():
            try:
                df = pd.read_parquet(path)
                frames.append(df)
            except Exception:
                continue
    
    if not frames:
        return pd.DataFrame()
    
    df = pd.concat(frames, ignore_index=True)
    df = df[df["ann_date"].astype(str) <= end_date]
    return df


# ==================== 回测引擎 ====================

class StrategyBacktester:
    """策略回测引擎"""

    def __init__(
        self,
        initial_capital: float = 1000000,
        top_n: int = 10,
        cost_rate: float = 0.003,
        data_delay_days: int = 2,
        enable_drawdown_control: bool = True,
        drawdown_reduce_threshold: float = 0.15,
        drawdown_clear_threshold: float = 0.25,
    ):
        self.initial_capital = initial_capital
        self.top_n = top_n
        self.cost_rate = cost_rate
        self.data_delay_days = data_delay_days
        self.enable_drawdown_control = enable_drawdown_control
        self.drawdown_reduce_threshold = drawdown_reduce_threshold
        self.drawdown_clear_threshold = drawdown_clear_threshold

    def run_backtest(
        self,
        trade_dates: List[str],
        daily_df: pd.DataFrame,
        daily_basic_df: pd.DataFrame,
        fina_df: pd.DataFrame,
        strategy_fn,
        strategy_name: str,
    ) -> Dict[str, Any]:
        """
        运行单策略回测

        Args:
            trade_dates: 交易日列表
            daily_df: 日线数据
            daily_basic_df: 每日基本面数据
            fina_df: 财务指标数据
            strategy_fn: 策略函数，输入(df, trade_date) -> signals DataFrame
            strategy_name: 策略名称

        Returns:
            回测结果字典
        """
        print(f"\n{'='*60}")
        print(f"回测策略: {strategy_name}")
        print(f"{'='*60}")

        portfolio_values = [self.initial_capital]
        daily_returns = []
        trade_records = []
        holdings = {}  # 当前持仓
        signal_cache = {}  # 信号缓存（用于 T+2 延迟）

        # 回撤控制变量
        max_portfolio_value = self.initial_capital
        position_ratio = 1.0  # 仓位比例（1.0=满仓）

        for i, trade_date in enumerate(trade_dates):
            if i % 20 == 0:
                print(f"  处理日期: {trade_date} ({i+1}/{len(trade_dates)})")

            # 获取当日数据
            day_mask = daily_df["trade_date"].astype(str) == trade_date
            day_data = daily_df[day_mask].copy()

            if day_data.empty:
                daily_returns.append(0.0)
                portfolio_values.append(portfolio_values[-1])
                continue

            # 生成信号（考虑 T+2 延迟）
            exec_date_idx = i + self.data_delay_days
            if exec_date_idx < len(trade_dates):
                exec_date = trade_dates[exec_date_idx]
                # 生成信号并缓存
                try:
                    signals = strategy_fn(day_data, trade_date, daily_basic_df, fina_df)
                    if signals is not None and not signals.empty:
                        signal_cache[exec_date] = signals
                except Exception as e:
                    pass  # 策略生成失败，保持空仓

            # 执行信号（T+2 后执行）
            if trade_date in signal_cache:
                signals = signal_cache[trade_date]

                # 回撤控制：动态调整仓位比例
                if self.enable_drawdown_control and portfolio_values:
                    current_value = portfolio_values[-1]
                    max_portfolio_value = max(max_portfolio_value, current_value)
                    current_drawdown = (current_value - max_portfolio_value) / max_portfolio_value

                    if current_drawdown < -self.drawdown_clear_threshold:
                        position_ratio = 0.0  # 清仓
                    elif current_drawdown < -self.drawdown_reduce_threshold:
                        position_ratio = 0.5  # 减仓50%
                    else:
                        position_ratio = 1.0  # 满仓

                # 计算换仓成本
                new_holdings = set(signals.nsmallest(self.top_n, "rank")["ts_code"].tolist())
                old_holdings = set(holdings.keys())

                # 换手率
                turnover = len(new_holdings.symmetric_difference(old_holdings)) / max(len(old_holdings), 1)
                cost = turnover * self.cost_rate

                # 更新持仓（应用仓位比例）
                if position_ratio > 0:
                    holdings = {code: position_ratio / self.top_n for code in new_holdings}
                else:
                    holdings = {}  # 清仓

                trade_records.append({
                    "trade_date": trade_date,
                    "signal_date": signal_cache[trade_date].iloc[0].get("trade_date", trade_date) if len(signal_cache[trade_date]) > 0 else trade_date,
                    "num_positions": len(holdings),
                    "turnover": turnover,
                    "cost": cost,
                    "position_ratio": position_ratio,
                    "current_drawdown": current_drawdown if self.enable_drawdown_control else 0.0,
                })

            # 计算当日收益
            if holdings:
                day_return = 0.0
                for code, weight in holdings.items():
                    code_data = day_data[day_data["ts_code"] == code]
                    if len(code_data) > 0:
                        prev_close = code_data.iloc[0].get("pre_close", code_data.iloc[0]["close"])
                        curr_close = code_data.iloc[0]["close"]
                        if prev_close and prev_close > 0:
                            ret = (curr_close - prev_close) / prev_close
                            day_return += weight * ret
                
                # 扣除交易成本
                if trade_records and trade_records[-1]["trade_date"] == trade_date:
                    day_return -= trade_records[-1]["cost"]
            else:
                day_return = 0.0

            daily_returns.append(day_return)
            portfolio_values.append(portfolio_values[-1] * (1 + day_return))

        # 计算评估指标
        metrics = self._calculate_metrics(daily_returns, portfolio_values)

        return {
            "strategy_name": strategy_name,
            "trade_dates": trade_dates,
            "daily_returns": daily_returns,
            "portfolio_values": portfolio_values,
            "metrics": metrics,
            "trade_records": trade_records,
        }

    def _calculate_metrics(
        self,
        daily_returns: List[float],
        portfolio_values: List[float],
    ) -> Dict[str, float]:
        """计算回测指标"""
        returns = pd.Series(daily_returns)
        values = pd.Series(portfolio_values)

        metrics = {}

        # 累计收益
        metrics["total_return"] = float(values.iloc[-1] / values.iloc[0] - 1)

        # 年化收益
        n_days = len(daily_returns)
        years = n_days / 252
        if years > 0 and values.iloc[-1] > 0:
            metrics["annual_return"] = float((values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1)
        else:
            metrics["annual_return"] = 0.0

        # 年化波动率
        if returns.std() > 0:
            metrics["annual_volatility"] = float(returns.std() * np.sqrt(252))
        else:
            metrics["annual_volatility"] = 0.0

        # 夏普比率
        risk_free_rate = 0.03
        if metrics["annual_volatility"] > 0:
            metrics["sharpe_ratio"] = float(
                (metrics["annual_return"] - risk_free_rate) / metrics["annual_volatility"]
            )
        else:
            metrics["sharpe_ratio"] = 0.0

        # 最大回撤
        cumulative = values / values.iloc[0]
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        metrics["max_drawdown"] = float(drawdown.min())

        # 胜率
        metrics["win_rate"] = float((returns > 0).mean())

        # 盈亏比
        winning = returns[returns > 0]
        losing = returns[returns < 0]
        if len(losing) > 0 and losing.mean() != 0:
            metrics["profit_loss_ratio"] = float(abs(winning.mean() / losing.mean()))
        else:
            metrics["profit_loss_ratio"] = float("inf") if len(winning) > 0 else 0.0

        # Calmar比率
        if metrics["max_drawdown"] != 0:
            metrics["calmar_ratio"] = float(metrics["annual_return"] / abs(metrics["max_drawdown"]))
        else:
            metrics["calmar_ratio"] = 0.0

        # 日均换手（从 trade_records 计算）
        metrics["avg_turnover"] = 0.0  # 需要从外部传入

        return metrics


# ==================== 策略函数 ====================

def create_champion_strategy_fn(
    daily_df: pd.DataFrame,
    daily_basic_df: pd.DataFrame,
    train_end: str,
    daily_dir: Path = None,
    daily_basic_path: Path = None,
):
    """创建 Champion 策略函数（预训练）"""
    import copy

    basic_cols = [c for c in ["ts_code", "trade_date", "pe_ttm", "pb", "turnover_rate", "total_mv", "circ_mv"]
                  if c in daily_basic_df.columns]

    # 加载训练期历史数据（train_end 前1年）
    train_frames = []
    if daily_dir is not None:
        train_start_year = max(2019, int(train_end[:4]) - 1)
        for year in range(train_start_year, int(train_end[:4]) + 1):
            p = daily_dir / f"daily_{year}.parquet"
            if p.exists():
                df_y = pd.read_parquet(p)
                train_frames.append(df_y[df_y["trade_date"].astype(str) < train_end])

    if train_frames:
        df_hist = pd.concat(train_frames, ignore_index=True)
        if daily_basic_path and daily_basic_path.exists():
            basic_hist = pd.read_parquet(daily_basic_path)
            basic_hist = basic_hist[basic_hist["trade_date"].astype(str) < train_end]
            hist_basic_cols = [c for c in basic_cols if c in basic_hist.columns]
            df_hist = df_hist.merge(basic_hist[hist_basic_cols], on=["ts_code", "trade_date"], how="left")
    else:
        df_hist = pd.DataFrame()

    # 回测期数据
    df_backtest = daily_df.merge(daily_basic_df[basic_cols], on=["ts_code", "trade_date"], how="left")
    df_all = pd.concat([df_hist, df_backtest], ignore_index=True) if not df_hist.empty else df_backtest

    # 训练数据
    df_train = df_all[df_all["trade_date"].astype(str) < train_end].copy()
    df_train["trade_date"] = pd.to_datetime(df_train["trade_date"], format="%Y%m%d", errors="coerce")

    # 训练模型
    selector_config = SelectionConfig(
        model_type="lgbm",
        label_horizons=(20, 60, 120),
        label_weights=(0.5, 0.3, 0.2),
        risk_adjusted=True,
        label_neutralized=True,
        ic_filter_enabled=False,
        industry_rank=False,
        exclude_bj=True,
        exclude_st=True,
    )
    print(f"[DEBUG] Champion cfg.industry_rank = {selector_config.industry_rank}")
    selector = StockSelector(selector_config)
    selector.fit(df_train)

    # 预计算回测期特征
    df_backtest["trade_date"] = pd.to_datetime(df_backtest["trade_date"], format="%Y%m%d", errors="coerce")
    df_prepared = selector.prepare_features(df_backtest)

    def strategy_fn(day_data: pd.DataFrame, trade_date: str, daily_basic: pd.DataFrame, fina: pd.DataFrame) -> pd.DataFrame:
        try:
            td = pd.to_datetime(trade_date)
            df_day = df_prepared[df_prepared["trade_date"] == td]

            if df_day.empty or len(df_day) < 50:
                return pd.DataFrame(columns=SIGNAL_SCHEMA)

            result = selector.predict_prepared(df_day)

            if result.empty:
                return pd.DataFrame(columns=SIGNAL_SCHEMA)

            # 格式化输出
            result = result.nlargest(30, "score").copy()
            result["trade_date"] = trade_date
            result["model_version"] = "seed_balance_strategy@v1.0.0"
            result["confidence"] = result.get("score", 0).abs()

            return result[["ts_code", "trade_date", "score", "rank", "confidence", "model_version"]]

        except Exception as e:
            return pd.DataFrame(columns=SIGNAL_SCHEMA)

    return strategy_fn, "seed_balance_strategy"


def create_challenger_strategy_fn(
    selector: MultiAlphaStockSelector,
    score_col: str,
    strategy_id: str,
):
    """创建 Challenger 策略函数"""

    def strategy_fn(day_data: pd.DataFrame, trade_date: str, daily_basic: pd.DataFrame, fina: pd.DataFrame) -> pd.DataFrame:
        try:
            result = selector.select(trade_date=trade_date, top_n=30)
            all_scores = result.get("all_scores", pd.DataFrame())
            
            if all_scores.empty or score_col not in all_scores.columns:
                return pd.DataFrame(columns=SIGNAL_SCHEMA)
            
            scores = all_scores[["ts_code", "trade_date", score_col]].copy()
            scores = scores.rename(columns={score_col: "score"})
            scores["rank"] = scores["score"].rank(ascending=False)
            scores["model_version"] = f"{strategy_id}@v1.0.0"
            scores["confidence"] = scores["score"].abs()
            
            return scores[["ts_code", "trade_date", "score", "rank", "confidence", "model_version"]]
            
        except Exception:
            return pd.DataFrame(columns=SIGNAL_SCHEMA)

    return strategy_fn, strategy_id


def _build_ma_regime(idx_df: pd.DataFrame) -> dict:
    """用 MA20/MA60 均线判断中期 regime"""
    df = idx_df.sort_values("date").copy()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    regime = np.where(
        (df["close"] > df["ma60"]) & (df["ma20"] > df["ma60"]), 2,
        np.where((df["close"] < df["ma60"]) & (df["ma20"] < df["ma60"]), 0, 1)
    )
    return dict(zip(df["date"].astype(str).str[:8].values, regime))


# 宏观 regime 标签（训练用）
_MACRO_REGIMES = [
    ("2019-01-01", "2021-02-10", "bull"),
    ("2021-02-11", "2022-04-26", "bear"),
    ("2022-04-27", "2022-07-05", "bull"),
    ("2022-07-06", "2024-09-23", "bear"),
    ("2024-09-24", "2024-10-08", "bull"),
    ("2024-10-09", "2024-11-04", "bear"),
    ("2024-11-05", "2025-03-14", "neutral"),
    ("2025-03-15", "2025-04-07", "bear"),
    ("2025-04-08", "2026-12-31", "bull"),
]


def _assign_macro_regime(dates: pd.Series) -> pd.Series:
    labels = pd.Series("neutral", index=dates.index)
    for start, end, regime in _MACRO_REGIMES:
        mask = (dates >= start) & (dates <= end)
        labels[mask] = regime
    return labels


def create_regime_strategy_fn(
    daily_df: pd.DataFrame,
    daily_basic_df: pd.DataFrame,
    idx_df: pd.DataFrame,
    train_end: str,
    daily_dir: Path = None,
    daily_basic_path: Path = None,
):
    """创建 Regime MA 策略（预训练+预计算特征）"""
    import copy

    basic_cols = [c for c in ["ts_code", "trade_date", "pe_ttm", "pb", "turnover_rate", "total_mv", "circ_mv"]
                  if c in daily_basic_df.columns]

    # 加载训练期历史数据（train_end 前3年，与之前报告一致）
    train_frames = []
    if daily_dir is not None:
        train_start_year = max(2019, int(train_end[:4]) - 3)
        for year in range(train_start_year, int(train_end[:4]) + 1):
            p = daily_dir / f"daily_{year}.parquet"
            if p.exists():
                df_y = pd.read_parquet(p)
                train_frames.append(df_y[df_y["trade_date"].astype(str) < train_end])

    if train_frames:
        df_hist = pd.concat(train_frames, ignore_index=True)
        if daily_basic_path and daily_basic_path.exists():
            basic_hist = pd.read_parquet(daily_basic_path)
            basic_hist = basic_hist[basic_hist["trade_date"].astype(str) < train_end]
            hist_basic_cols = [c for c in basic_cols if c in basic_hist.columns]
            df_hist = df_hist.merge(basic_hist[hist_basic_cols], on=["ts_code", "trade_date"], how="left")
    else:
        df_hist = pd.DataFrame()

    # 回测期数据（用于推理）
    df_backtest = daily_df.merge(daily_basic_df[basic_cols], on=["ts_code", "trade_date"], how="left")

    df_all = pd.concat([df_hist, df_backtest], ignore_index=True) if not df_hist.empty else df_backtest

    # 训练数据
    df_train = df_all[df_all["trade_date"].astype(str) < train_end].copy()
    df_train["trade_date"] = pd.to_datetime(df_train["trade_date"], format="%Y%m%d", errors="coerce")
    df_train["regime"] = _assign_macro_regime(df_train["trade_date"])

    cfg = SelectionConfig(
        model_type="lgbm", label_neutralized=True,
        ic_filter_enabled=False,  # 禁用IC过滤，加速训练
        industry_rank=False,  # 禁用行业排名，加速训练
        exclude_bj=True, exclude_st=True,
    )
    print(f"[DEBUG] cfg.industry_rank = {cfg.industry_rank}")
    regime_cfg = RegimeSelectionConfig(base_config=copy.deepcopy(cfg))
    print(f"[DEBUG] regime_cfg.base_config.industry_rank = {regime_cfg.base_config.industry_rank}")
    regime_model = RegimeStockSelector(regime_cfg)
    regime_model.fit(df_train, df_train["regime"])

    # 预计算全量特征（仅回测期）
    df_backtest["trade_date"] = pd.to_datetime(df_backtest["trade_date"], format="%Y%m%d", errors="coerce")
    df_prepared = regime_model.fallback_model.prepare_features(df_backtest)

    # MA regime
    idx_df = idx_df.copy()
    idx_df["date"] = pd.to_datetime(idx_df["date"])
    d2s_ma = {}
    ma20 = idx_df["close"].rolling(20).mean()
    ma60 = idx_df["close"].rolling(60).mean()
    for i, row in idx_df.iterrows():
        r = 2 if (row["close"] > ma60.iloc[i] and ma20.iloc[i] > ma60.iloc[i]) else (
            0 if (row["close"] < ma60.iloc[i] and ma20.iloc[i] < ma60.iloc[i]) else 1)
        d2s_ma[row["date"].strftime("%Y%m%d")] = r

    # 缓存预测结果
    pred_cache = {}

    def strategy_fn(day_data, trade_date, daily_basic, fina):
        try:
            td = pd.to_datetime(trade_date)
            df_day = df_prepared[df_prepared["trade_date"] == td]
            if len(df_day) < 50:
                return pd.DataFrame(columns=SIGNAL_SCHEMA)
            regime = d2s_ma.get(str(trade_date)[:8], 1)
            pred = regime_model.predict_prepared(df_day, regime)
            result = pred.nlargest(30, "score").copy()
            result["trade_date"] = trade_date
            result["model_version"] = "regime_ma_strategy@v1.0.0"
            result["confidence"] = result["score"].abs()
            result["rank"] = range(1, len(result) + 1)
            return result[["ts_code", "trade_date", "score", "rank", "confidence", "model_version"]]
        except Exception:
            return pd.DataFrame(columns=SIGNAL_SCHEMA)

    return strategy_fn, "regime_ma_strategy"


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="Champion/Challenger 回测验证")
    parser.add_argument("--start-date", type=str, default="20230601", help="回测开始日期")
    parser.add_argument("--end-date", type=str, default="20241231", help="回测结束日期")
    parser.add_argument("--top-n", type=int, default=10, help="TopN 选股数量")
    parser.add_argument("--initial-capital", type=float, default=1000000, help="初始资金")
    parser.add_argument("--cost-rate", type=float, default=0.003, help="交易成本率")
    parser.add_argument("--data-dir", type=str, default=None, help="Tushare 数据根目录")
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="输出目录，默认 data/backtest/champion_challenger",
    )
    parser.add_argument("--strategies", type=str, default="regime,champion,balance,positive,value,satellite", help="策略列表")
    args = parser.parse_args()

    # 解析策略列表
    strategy_names = [s.strip().lower() for s in args.strategies.split(",")]
    
    # 数据目录
    data_root = _resolve_tushare_root(args.data_dir)
    daily_dir = data_root / "daily"
    daily_basic_path = data_root / "daily_basic_all.parquet"
    fina_dir = data_root / "fundamental"

    print("=" * 80)
    print("Champion/Challenger 回测验证")
    print("=" * 80)
    print(f"回测区间: {args.start_date} ~ {args.end_date}")
    print(f"策略列表: {strategy_names}")
    print(f"数据目录: {data_root}")

    # 加载数据
    print("\n加载数据...")
    trade_dates = _find_trade_dates(daily_dir, args.start_date, args.end_date)
    if not trade_dates:
        raise ValueError("未找到交易日数据")

    print(f"交易日数量: {len(trade_dates)}")
    
    daily_df = _load_daily_data(daily_dir, trade_dates)
    print(f"日线数据: {len(daily_df)} 条")

    daily_basic_df = _load_daily_basic(daily_basic_path, trade_dates)
    print(f"每日基本面: {len(daily_basic_df)} 条")

    fina_df = _load_fina_indicator(fina_dir, args.end_date)
    print(f"财务指标: {len(fina_df)} 条")

    # 创建回测引擎
    backtester = StrategyBacktester(
        initial_capital=args.initial_capital,
        top_n=args.top_n,
        cost_rate=args.cost_rate,
    )

    # 运行各策略回测
    results = []

    # Champion 策略
    if "champion" in strategy_names:
        print("\n准备 Champion 策略...")
        strategy_fn, strategy_id = create_champion_strategy_fn(
            daily_df=daily_df,
            daily_basic_df=daily_basic_df,
            train_end=args.start_date,
            daily_dir=daily_dir,
            daily_basic_path=daily_basic_path,
        )
        result = backtester.run_backtest(
            trade_dates=trade_dates,
            daily_df=daily_df,
            daily_basic_df=daily_basic_df,
            fina_df=fina_df,
            strategy_fn=strategy_fn,
            strategy_name=strategy_id,
        )
        results.append(result)

    # Regime MA 策略（新Champion）
    if "regime" in strategy_names:
        print("\n准备 Regime MA 策略...")
        idx_path = data_root / "index" / "index_000300_SH_ohlc.parquet"
        if idx_path.exists():
            idx_df = pd.read_parquet(idx_path)
            strategy_fn, strategy_id = create_regime_strategy_fn(
                daily_df=daily_df, daily_basic_df=daily_basic_df,
                idx_df=idx_df, train_end=args.start_date,
                daily_dir=daily_dir, daily_basic_path=daily_basic_path,
            )
            result = backtester.run_backtest(
                trade_dates=trade_dates,
                daily_df=daily_df,
                daily_basic_df=daily_basic_df,
                fina_df=fina_df,
                strategy_fn=strategy_fn,
                strategy_name=strategy_id,
            )
            results.append(result)
        else:
            print("  [跳过] 缺少沪深300指数数据")

    # Challenger 策略
    challenger_map = {
        "balance": ("combined_score", "balance_strategy_v1"),
        "positive": ("positive_score", "positive_strategy_v1"),
        "value": ("value_score", "value_strategy_v1"),
        "satellite": ("satellite_score", "satellite_strategy_v1"),
    }

    if any(name in strategy_names for name in challenger_map.keys()):
        print("\n准备 Challenger 策略...")
        selector = MultiAlphaStockSelector(data_dir=str(data_root))
        
        for name, (score_col, strategy_id) in challenger_map.items():
            if name in strategy_names:
                strategy_fn, sid = create_challenger_strategy_fn(selector, score_col, strategy_id)
                result = backtester.run_backtest(
                    trade_dates=trade_dates,
                    daily_df=daily_df,
                    daily_basic_df=daily_basic_df,
                    fina_df=fina_df,
                    strategy_fn=strategy_fn,
                    strategy_name=sid,
                )
                results.append(result)

    # 生成评估报告
    print("\n" + "=" * 80)
    print("回测结果汇总")
    print("=" * 80)

    summary_rows = []
    for result in results:
        metrics = result["metrics"]
        row = {
            "strategy_name": result["strategy_name"],
            **metrics,
        }
        summary_rows.append(row)
        print(f"\n{result['strategy_name']}:")
        print(f"  累计收益: {metrics['total_return']:.2%}")
        print(f"  年化收益: {metrics['annual_return']:.2%}")
        print(f"  年化波动: {metrics['annual_volatility']:.2%}")
        print(f"  夏普比率: {metrics['sharpe_ratio']:.2f}")
        print(f"  最大回撤: {metrics['max_drawdown']:.2%}")
        print(f"  胜率: {metrics['win_rate']:.2%}")
        print(f"  Calmar比率: {metrics['calmar_ratio']:.2f}")

    # 保存结果
    output_root = Path(args.output_root) if args.output_root else (get_data_root() / "backtest" / "champion_challenger")
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存汇总结果
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_root / f"backtest_summary_{run_id}.parquet"
    summary_df.to_parquet(summary_path, index=False)
    
    # 保存详细结果
    detail_path = output_root / f"backtest_detail_{run_id}.json"
    detail = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_id,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "top_n": args.top_n,
        "initial_capital": args.initial_capital,
        "cost_rate": args.cost_rate,
        "trade_dates_count": len(trade_dates),
        "summary": summary_rows,
    }
    detail_path.write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")

    # 更新最新结果
    latest_path = output_root / "backtest_summary_latest.parquet"
    summary_df.to_parquet(latest_path, index=False)
    
    latest_json = output_root / "backtest_summary_latest.json"
    latest_json.write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n回测结果已保存: {output_root}")
    print(f"汇总文件: {summary_path}")
    print(f"详细文件: {detail_path}")

    # 确定最佳策略
    best_strategy = max(summary_rows, key=lambda x: x.get("sharpe_ratio", 0))
    print(f"\n最佳策略 (按夏普比率): {best_strategy['strategy_name']}")
    print(f"  夏普比率: {best_strategy['sharpe_ratio']:.2f}")
    print(f"  年化收益: {best_strategy['annual_return']:.2%}")


if __name__ == "__main__":
    main()
