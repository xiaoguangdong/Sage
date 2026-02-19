#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
行业景气度：个股信号聚合到申万一级

依赖：
1) 财务数据：tushare/fundamental/fina_indicator_*.parquet
2) 行业成分：tushare/northbound/sw_constituents.parquet

输出：
processed/industry_prosperity_sw_l1.parquet
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from sage_core.industry.signal_indicators import IndustryProsperity
from scripts.data._shared.runtime import get_data_path, get_tushare_root


def load_constituents(tushare_root: Path) -> pd.DataFrame:
    path = tushare_root / "northbound" / "sw_constituents.parquet"
    if not path.exists():
        raise FileNotFoundError(f"缺少行业成分文件: {path}")
    df = pd.read_parquet(path)
    df = df.rename(columns={"con_code": "ts_code"})
    df["in_date"] = pd.to_datetime(df["in_date"])
    df["out_date"] = pd.to_datetime(df["out_date"])
    df["out_date"] = df["out_date"].fillna(pd.Timestamp("2099-12-31"))
    return df


def aggregate_prosperity(
    tushare_root: Path,
    output_dir: Path,
    epsilon: float = 0.01,
) -> Path:
    prosperity = IndustryProsperity(data_dir=tushare_root / "fundamental")
    df = prosperity.get_prosperity_signal(epsilon=epsilon)
    if df is None or df.empty:
        raise RuntimeError("行业景气度数据为空")

    df["end_date"] = pd.to_datetime(df["end_date"])
    const = load_constituents(tushare_root)

    merged = df.merge(
        const[["ts_code", "industry_name", "in_date", "out_date"]],
        on="ts_code",
        how="left",
    )
    valid = merged[(merged["end_date"] >= merged["in_date"]) & (merged["end_date"] <= merged["out_date"])].copy()

    if valid.empty:
        raise RuntimeError("无法匹配行业成分（无有效记录）")

    agg = (
        valid.groupby(["industry_name", "end_date"])
        .agg(
            prosperity_ratio=("prosperity_signal", "mean"),
            revenue_acceleration_mean=("revenue_acceleration", "mean"),
            gross_margin_change_mean=("margin_change", "mean"),
            sample_count=("ts_code", "count"),
        )
        .reset_index()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "industry_prosperity_sw_l1.parquet"
    agg.to_parquet(output_path, index=False)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tushare-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epsilon", type=float, default=0.01)
    args = parser.parse_args()

    tushare_root = Path(args.tushare_root) if args.tushare_root else get_tushare_root()
    output_dir = Path(args.output_dir) if args.output_dir else get_data_path("processed")

    output_path = aggregate_prosperity(tushare_root, output_dir, epsilon=args.epsilon)
    print(f"行业景气度聚合完成: {output_path}")


if __name__ == "__main__":
    main()
