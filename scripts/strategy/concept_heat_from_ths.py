#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于同花顺板块指数(ths_daily)计算概念/板块热度因子
输出：data/processed/concepts/concept_heat.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data._shared.runtime import get_data_root, get_data_path


def build_heat(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["ts_code", "trade_date"])

    df["ret_20d"] = df.groupby("ts_code")["close"].transform(lambda s: s / s.shift(20) - 1)
    df["ret_60d"] = df.groupby("ts_code")["close"].transform(lambda s: s / s.shift(60) - 1)
    df["turnover_20d"] = df.groupby("ts_code")["turnover_rate"].transform(lambda s: s.rolling(20).mean())
    df["vol_20d"] = df.groupby("ts_code")["pct_change"].transform(lambda s: s.rolling(20).std())

    # 简单热度分：动量 + 换手 + 波动（可后续配置权重）
    df["heat_score"] = (
        df["ret_20d"].fillna(0) * 0.5
        + df["ret_60d"].fillna(0) * 0.3
        + df["turnover_20d"].fillna(0) * 0.2 / 100
    )

    keep = [
        "ts_code",
        "trade_date",
        "ret_20d",
        "ret_60d",
        "turnover_20d",
        "vol_20d",
        "heat_score",
    ]
    return df[keep]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    input_path = Path(args.input_path) if args.input_path else get_data_root() / "tushare" / "concepts" / "ths_daily.parquet"
    if not input_path.exists():
        print(f"未找到 ths_daily: {input_path}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else get_data_path("processed", "concepts", ensure=True)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    if df.empty:
        print("ths_daily 为空")
        return

    heat = build_heat(df)
    output_path = output_dir / "concept_heat.parquet"
    heat.to_parquet(output_path, index=False)
    print(f"已保存: {output_path}")


if __name__ == "__main__":
    main()
