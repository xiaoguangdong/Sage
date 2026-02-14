#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
政策信号回测（MVP）

流程：
1) 读取行业级政策信号（policy_signals.parquet）
2) 映射到股票（行业 -> 股票）
3) 生成交易信号并接入 SimpleBacktestEngine
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from sage_core.backtest.simple_engine import SimpleBacktestEngine
from sage_core.backtest.types import BacktestConfig
from scripts.data._shared.runtime import get_data_path
from scripts.data.macro.clean_macro_data import MacroDataProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_policy_path(path: Optional[str]) -> Path:
    if path:
        p = Path(path)
        return p if p.is_absolute() else PROJECT_ROOT / p
    return get_data_path("processed", "policy") / "policy_signals.parquet"


def _load_policy_signals(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"政策信号文件不存在: {path}")
    df = pd.read_parquet(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y%m%d")
    return df


def _load_daily_returns(start: str, end: str) -> pd.DataFrame:
    daily_dir = PROJECT_ROOT / "data" / "tushare" / "daily"
    if not daily_dir.exists():
        raise FileNotFoundError("未找到日线数据目录 data/tushare/daily")

    years = range(int(start[:4]), int(end[:4]) + 1)
    frames = []
    for y in years:
        path = daily_dir / f"daily_{y}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        frames.append(df)

    if not frames:
        raise FileNotFoundError("未加载到任何日线数据")

    data = pd.concat(frames, ignore_index=True)
    data["trade_date"] = data["trade_date"].astype(str)
    data = data[(data["trade_date"] >= start) & (data["trade_date"] <= end)]
    data["ret"] = pd.to_numeric(data["pct_chg"], errors="coerce") / 100.0
    return data[["trade_date", "ts_code", "ret"]]


def _load_hs300_constituents() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "tushare" / "constituents" / "hs300_constituents_all.parquet"
    if not path.exists():
        raise FileNotFoundError("未找到 hs300_constituents_all.parquet")
    df = pd.read_parquet(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def _select_members_for_date(hs300: pd.DataFrame, trade_date: pd.Timestamp) -> List[str]:
    sub = hs300[hs300["trade_date"] <= trade_date]
    if sub.empty:
        return []
    latest_date = sub["trade_date"].max()
    return sub[sub["trade_date"] == latest_date]["con_code"].tolist()


def build_stock_signals(
    policy: pd.DataFrame,
    industry_map: pd.DataFrame,
    hs300: pd.DataFrame,
    top_industries: int = 5,
) -> pd.DataFrame:
    signals = []
    policy = policy.sort_values("trade_date")
    industry_map = industry_map.rename(columns={"sw_industry": "industry"})

    for date_str, day in policy.groupby("trade_date"):
        day = day.sort_values("policy_score", ascending=False).head(top_industries)
        trade_date = pd.to_datetime(date_str)
        members = set(_select_members_for_date(hs300, trade_date))
        if not members:
            continue
        merged = day.merge(industry_map, on="industry", how="left")
        merged = merged[merged["ts_code"].isin(members)]
        if merged.empty:
            continue
        for _, row in merged.iterrows():
            signals.append({
                "trade_date": date_str,
                "ts_code": row["ts_code"],
                "score": row["policy_score"],
                "industry": row["industry"],
            })
    return pd.DataFrame(signals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", type=str, default=None, help="政策信号文件路径")
    parser.add_argument("--top-industries", type=int, default=5, help="每日取政策分数最高的行业数")
    parser.add_argument("--start-date", type=str, default=None, help="回测开始日期YYYYMMDD")
    parser.add_argument("--end-date", type=str, default=None, help="回测结束日期YYYYMMDD")
    parser.add_argument("--cost-rate", type=float, default=0.005, help="单边交易成本(千分之5=0.005)")
    parser.add_argument("--data-delay-days", type=int, default=2, help="信号生效延迟天数")
    args = parser.parse_args()

    policy_path = _resolve_policy_path(args.policy_path)
    policy = _load_policy_signals(policy_path)
    if policy.empty:
        raise ValueError("政策信号为空")

    start = args.start_date or policy["trade_date"].min()
    end = args.end_date or policy["trade_date"].max()

    hs300 = _load_hs300_constituents()
    industry_map = MacroDataProcessor().load_sw_stock_industry_map()
    if industry_map.empty:
        raise ValueError("未加载到股票行业映射")

    signals = build_stock_signals(policy, industry_map, hs300, top_industries=args.top_industries)
    if signals.empty:
        raise ValueError("未生成任何股票信号")

    returns = _load_daily_returns(start, end)
    config = BacktestConfig(
        cost_rate=args.cost_rate,
        data_delay_days=args.data_delay_days,
        max_positions=10,
        max_industry_weight=0.40,
        t_plus_one=True,
    )

    engine = SimpleBacktestEngine(config)
    result = engine.run(signals, returns)

    output_dir = get_data_path("processed", "backtest", ensure=True)
    metrics_path = output_dir / "policy_backtest_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result.metrics, f, ensure_ascii=False, indent=2)
    print(f"回测完成，指标已保存: {metrics_path}")


if __name__ == "__main__":
    main()
