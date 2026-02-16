#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from sage_core.industry.industry_backtest_eval import (
    prepare_industry_returns,
    prepare_prosperity_scores,
    prepare_trend_dominant_state,
    prepare_trend_gate,
)
from scripts.data._shared.runtime import get_data_path, get_tushare_root


def _resolve_path(raw: str | None, default: Path) -> Path:
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else PROJECT_ROOT / path
    return default


def _normalize_trade_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y%m%d")


def _compound_return(values: pd.Series) -> float:
    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.empty:
        return np.nan
    return float(np.prod(1.0 + series.values) - 1.0)


def _future_returns(industry_returns: pd.DataFrame, hold_days: int) -> pd.DataFrame:
    source = industry_returns.copy()
    source["trade_date"] = _normalize_trade_date(source["trade_date"])
    source = source.dropna(subset=["trade_date", "sw_industry", "industry_return"])
    source["industry_return"] = pd.to_numeric(source["industry_return"], errors="coerce")
    source = source.dropna(subset=["industry_return"])

    trade_dates = sorted(source["trade_date"].unique().tolist())
    index_map = {date: idx for idx, date in enumerate(trade_dates)}
    rows: List[pd.DataFrame] = []
    for date in trade_dates:
        start_idx = index_map[date] + 1
        end_idx = start_idx + int(hold_days)
        if end_idx > len(trade_dates):
            continue
        hold = trade_dates[start_idx:end_idx]
        block = source[source["trade_date"].isin(hold)]
        grouped = block.groupby("sw_industry", as_index=False)["industry_return"].apply(_compound_return)
        grouped = grouped.rename(columns={"industry_return": "future_return"})
        grouped["trade_date"] = date
        rows.append(grouped)
    if not rows:
        return pd.DataFrame(columns=["trade_date", "sw_industry", "future_return"])
    return pd.concat(rows, ignore_index=True)


def _ic_by_date(scores: pd.DataFrame, future_returns: pd.DataFrame, rebalance_step: int) -> pd.DataFrame:
    source = scores[["trade_date", "sw_industry", "score"]].merge(
        future_returns,
        on=["trade_date", "sw_industry"],
        how="inner",
    )
    source = source.dropna(subset=["trade_date", "score", "future_return"])
    if source.empty:
        return pd.DataFrame(columns=["trade_date", "ic"])

    trade_dates = sorted(source["trade_date"].unique().tolist())
    selected_dates = trade_dates[:: max(1, int(rebalance_step))]
    source = source[source["trade_date"].isin(selected_dates)].copy()

    rows = []
    for date, frame in source.groupby("trade_date"):
        if len(frame) < 5:
            rows.append({"trade_date": date, "ic": np.nan, "sample_size": int(len(frame))})
            continue
        ic = frame["score"].rank(pct=True).corr(frame["future_return"].rank(pct=True), method="pearson")
        rows.append({"trade_date": date, "ic": float(ic) if pd.notna(ic) else np.nan, "sample_size": int(len(frame))})
    return pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)


def _summary(ic_series: pd.Series, hold_days: int) -> Dict[str, float]:
    series = pd.to_numeric(ic_series, errors="coerce").dropna()
    if series.empty:
        return {
            "count": 0,
            "ic_mean": 0.0,
            "ic_std": 0.0,
            "ic_ir": 0.0,
            "ic_positive_ratio": 0.0,
        }
    mean = float(series.mean())
    std = float(series.std(ddof=0))
    if std > 0:
        periods_per_year = 252.0 / max(1, int(hold_days))
        ir = float(mean / std * np.sqrt(periods_per_year))
    else:
        ir = 0.0
    return {
        "count": int(len(series)),
        "ic_mean": mean,
        "ic_std": std,
        "ic_ir": ir,
        "ic_positive_ratio": float((series > 0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="景气评分分状态IC诊断")
    parser.add_argument("--sw-daily-path", type=str, default=None)
    parser.add_argument("--sw-l1-map-path", type=str, default=None)
    parser.add_argument("--benchmark-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--hold-days", type=int, default=5)
    parser.add_argument("--rebalance-step", type=int, default=5)
    parser.add_argument("--trend-neutral-multiplier", type=float, default=0.6)
    parser.add_argument("--trend-risk-off-multiplier", type=float, default=0.2)
    parser.add_argument("--trend-dominance-lookback-days", type=int, default=5)
    parser.add_argument("--trend-dominance-threshold", type=float, default=0.6)
    parser.add_argument("--prosperity-momentum-window", type=int, default=20)
    parser.add_argument("--prosperity-amount-window", type=int, default=20)
    parser.add_argument("--prosperity-volatility-window", type=int, default=20)
    args = parser.parse_args()

    tushare_root = get_tushare_root()
    sw_daily_path = _resolve_path(
        args.sw_daily_path,
        tushare_root / "sectors" / "sw_daily_all.parquet",
    )
    sw_l1_map_path = _resolve_path(
        args.sw_l1_map_path,
        tushare_root / "sw_industry" / "sw_industry_l1.parquet",
    )
    benchmark_path = _resolve_path(
        args.benchmark_path,
        tushare_root / "index" / "index_000300_SH_ohlc.parquet",
    )
    output_dir = _resolve_path(
        args.output_dir,
        get_data_path("backtest", "industry", "diagnostics", ensure=True),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not sw_daily_path.exists():
        raise FileNotFoundError(f"未找到申万行业日线: {sw_daily_path}")
    if not sw_l1_map_path.exists():
        raise FileNotFoundError(f"未找到申万L1映射: {sw_l1_map_path}")
    if not benchmark_path.exists():
        raise FileNotFoundError(f"未找到基准指数: {benchmark_path}")

    sw_daily = pd.read_parquet(sw_daily_path)
    sw_l1_map = pd.read_parquet(sw_l1_map_path)
    benchmark_df = pd.read_parquet(benchmark_path)

    prosperity_scores = prepare_prosperity_scores(
        sw_daily,
        sw_l1_map,
        momentum_window=args.prosperity_momentum_window,
        amount_window=args.prosperity_amount_window,
        volatility_window=args.prosperity_volatility_window,
    )
    industry_returns = prepare_industry_returns(sw_daily, sw_l1_map)
    future_returns = _future_returns(industry_returns, hold_days=args.hold_days)
    ic_daily = _ic_by_date(prosperity_scores, future_returns, rebalance_step=args.rebalance_step)

    trend_gate = prepare_trend_gate(
        benchmark_df,
        neutral_multiplier=args.trend_neutral_multiplier,
        risk_off_multiplier=args.trend_risk_off_multiplier,
    )
    dominant_state = prepare_trend_dominant_state(
        trend_gate,
        lookback_days=args.trend_dominance_lookback_days,
        dominance_threshold=args.trend_dominance_threshold,
    )
    ic_daily = ic_daily.merge(dominant_state[["trade_date", "dominant_state"]], on="trade_date", how="left")

    summary = {
        "overall": _summary(ic_daily["ic"], hold_days=args.hold_days),
        "by_state": {},
        "params": {
            "hold_days": int(args.hold_days),
            "rebalance_step": int(args.rebalance_step),
            "trend_dominance_lookback_days": int(args.trend_dominance_lookback_days),
            "trend_dominance_threshold": float(args.trend_dominance_threshold),
            "prosperity_momentum_window": int(args.prosperity_momentum_window),
            "prosperity_amount_window": int(args.prosperity_amount_window),
            "prosperity_volatility_window": int(args.prosperity_volatility_window),
        },
    }
    for state in ["RISK_ON", "NEUTRAL", "RISK_OFF"]:
        summary["by_state"][state] = _summary(
            ic_daily.loc[ic_daily["dominant_state"] == state, "ic"],
            hold_days=args.hold_days,
        )

    ic_path = output_dir / "prosperity_ic_by_date.parquet"
    summary_path = output_dir / "prosperity_ic_by_state_summary.json"
    ic_daily.to_parquet(ic_path, index=False)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"已输出: {ic_path}")
    print(f"已输出: {summary_path}")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["by_state"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
