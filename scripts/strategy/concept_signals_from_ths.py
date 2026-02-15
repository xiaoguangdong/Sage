#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于同花顺概念指数(ths_daily)生成概念信号
输出：data/signals/concept_signals.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data._shared.runtime import get_data_path, get_tushare_root


def load_ths_daily(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"未找到 ths_daily: {input_path}")
    df = pd.read_parquet(input_path)
    if df.empty:
        raise ValueError("ths_daily 为空")
    return df


def load_ths_index(index_path: Path) -> pd.DataFrame:
    if not index_path.exists():
        return pd.DataFrame(columns=["ts_code", "name"])
    df = pd.read_parquet(index_path)
    cols = df.columns.tolist()
    name_col = "name" if "name" in cols else ("index_name" if "index_name" in cols else None)
    if name_col:
        df = df.rename(columns={name_col: "name"})
    return df[["ts_code", "name"]].drop_duplicates()


def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["ts_code", "trade_date"])

    df["ret_20d"] = df.groupby("ts_code")["close"].transform(lambda s: s / s.shift(20) - 1)
    df["ret_60d"] = df.groupby("ts_code")["close"].transform(lambda s: s / s.shift(60) - 1)
    df["turnover_20d"] = df.groupby("ts_code")["turnover_rate"].transform(lambda s: s.rolling(20).mean())
    df["vol_20d"] = df.groupby("ts_code")["pct_change"].transform(lambda s: s.rolling(20).std())

    df["heat_score"] = (
        df["ret_20d"].fillna(0) * 0.5
        + df["ret_60d"].fillna(0) * 0.3
        + df["turnover_20d"].fillna(0) * 0.2 / 100
    )

    df["rank_pct"] = df.groupby("trade_date")["heat_score"].rank(pct=True)
    df["overheat_flag"] = (df["rank_pct"] >= 0.95).astype(int)

    keep = [
        "ts_code",
        "trade_date",
        "ret_20d",
        "ret_60d",
        "turnover_20d",
        "vol_20d",
        "heat_score",
        "overheat_flag",
    ]
    return df[keep]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, default=None)
    parser.add_argument("--index-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=None, help="仅保留每个交易日热度TopK概念")
    args = parser.parse_args()

    tushare_root = get_tushare_root()
    input_path = Path(args.input_path) if args.input_path else tushare_root / "concepts" / "ths_daily.parquet"
    index_path = Path(args.index_path) if args.index_path else tushare_root / "concepts" / "ths_index.parquet"
    output_dir = Path(args.output_dir) if args.output_dir else get_data_path("signals", ensure=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_ths_daily(input_path)
    signals = build_signals(df)

    names = load_ths_index(index_path)
    if not names.empty:
        signals = signals.merge(names, on="ts_code", how="left")

    signals = signals.rename(columns={
        "ts_code": "concept_code",
        "name": "concept_name",
        "ret_20d": "heat_20d",
        "ret_60d": "heat_60d",
        "heat_score": "concept_heat_score",
        "turnover_20d": "concept_turnover_20d",
        "vol_20d": "concept_vol_20d",
    })

    if args.top_k:
        signals = signals.sort_values(["trade_date", "concept_heat_score"], ascending=[True, False])
        signals = signals.groupby("trade_date").head(args.top_k).reset_index(drop=True)
        output_path = output_dir / f"concept_signals_top{args.top_k}.parquet"
    else:
        output_path = output_dir / "concept_signals.parquet"

    signals.to_parquet(output_path, index=False)
    print(f"已保存: {output_path}")


if __name__ == "__main__":
    main()
