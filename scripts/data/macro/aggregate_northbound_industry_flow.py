#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
北向资金行业配置比例聚合

输入:
- tushare/northbound/hk_hold.parquet
- tushare/northbound/sw_constituents.parquet

输出:
- tushare/northbound/industry_northbound_flow.parquet
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data._shared.runtime import get_tushare_root


def load_inputs(tushare_root: Path):
    hk_path = tushare_root / "northbound" / "hk_hold.parquet"
    const_path = tushare_root / "northbound" / "sw_constituents.parquet"
    if not hk_path.exists():
        raise FileNotFoundError(f"缺少北向持仓: {hk_path}")
    if not const_path.exists():
        raise FileNotFoundError(f"缺少行业成分: {const_path}")
    hk = pd.read_parquet(hk_path)
    const = pd.read_parquet(const_path).rename(columns={"con_code": "ts_code"})
    const["in_date"] = pd.to_datetime(const["in_date"])
    const["out_date"] = pd.to_datetime(const["out_date"])
    const["out_date"] = const["out_date"].fillna(pd.Timestamp("2099-12-31"))
    return hk, const


def aggregate(tushare_root: Path) -> Path:
    hk, const = load_inputs(tushare_root)
    hk["trade_date"] = pd.to_datetime(hk["trade_date"])
    hk["vol"] = pd.to_numeric(hk["vol"], errors="coerce")
    hk["ratio"] = pd.to_numeric(hk["ratio"], errors="coerce")

    merged = hk.merge(
        const[["ts_code", "industry_code", "industry_name", "in_date", "out_date"]],
        on="ts_code",
        how="left",
    )
    valid = merged[
        (merged["trade_date"] >= merged["in_date"]) & (merged["trade_date"] <= merged["out_date"])
    ].copy()

    if valid.empty:
        raise RuntimeError("无法匹配行业成分（无有效记录）")

    agg = valid.groupby(["industry_code", "industry_name", "trade_date"]).agg(
        vol=("vol", "sum"),
        ratio=("ratio", "mean"),
    ).reset_index()

    total_vol = agg.groupby("trade_date")["vol"].transform("sum")
    agg["industry_ratio"] = agg["vol"] / total_vol.replace(0, pd.NA)

    output_path = tushare_root / "northbound" / "industry_northbound_flow.parquet"
    agg.to_parquet(output_path, index=False)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tushare-root", default=None)
    args = parser.parse_args()

    tushare_root = Path(args.tushare_root) if args.tushare_root else get_tushare_root()
    output_path = aggregate(tushare_root)
    print(f"北向行业配置已生成: {output_path}")


if __name__ == "__main__":
    main()
