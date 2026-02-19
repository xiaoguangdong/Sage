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


def _looks_like_a_share(ts_code: str) -> bool:
    code = str(ts_code).upper()
    return code.endswith((".SH", ".SZ", ".BJ"))


def _build_sw_constituents_from_sw_industry(tushare_root: Path) -> pd.DataFrame:
    l1_path = tushare_root / "sw_industry" / "sw_industry_l1.parquet"
    member_path = tushare_root / "sw_industry" / "sw_index_member.parquet"
    if not l1_path.exists() or not member_path.exists():
        raise FileNotFoundError(f"缺少申万行业数据: {l1_path} / {member_path}")

    l1 = pd.read_parquet(l1_path).copy()
    members = pd.read_parquet(member_path).copy()
    name_col = "industry_name" if "industry_name" in l1.columns else "index_name"
    l1 = l1.rename(columns={name_col: "industry_name"})
    members = members.rename(columns={"con_code": "ts_code"})
    keep_cols = [c for c in ["index_code", "industry_code", "industry_name"] if c in l1.columns]
    if "index_code" not in keep_cols:
        raise ValueError(f"sw_industry_l1 缺少 index_code: {l1.columns.tolist()}")
    merged = members.merge(l1[keep_cols], on="index_code", how="left")
    merged = merged.dropna(subset=["ts_code", "industry_name"]).copy()
    merged["ts_code"] = merged["ts_code"].astype(str)
    merged = merged[merged["ts_code"].map(_looks_like_a_share)]
    merged = merged.drop_duplicates(
        subset=["ts_code", "industry_name", "index_code", "in_date", "out_date"], keep="first"
    )
    return merged


def _select_holding_source(tushare_root: Path) -> Path:
    candidates = [
        tushare_root / "northbound" / "hk_hold.parquet",
        tushare_root / "northbound" / "northbound_hold.parquet",
        tushare_root / "northbound" / "northbound_hk_hold.parquet",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            sample = pd.read_parquet(path, columns=["ts_code"]).head(200)
        except Exception:
            continue
        if sample.empty:
            continue
        ratio = sample["ts_code"].astype(str).map(_looks_like_a_share).mean()
        if ratio >= 0.6:
            return path
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"缺少北向持仓文件: {candidates}")


def load_inputs(tushare_root: Path):
    hk_path = _select_holding_source(tushare_root)
    const_path = tushare_root / "northbound" / "sw_constituents.parquet"
    if not const_path.exists():
        const = _build_sw_constituents_from_sw_industry(tushare_root)
        const_path.parent.mkdir(parents=True, exist_ok=True)
        const.to_parquet(const_path, index=False)
    else:
        const = pd.read_parquet(const_path)
        industries = const["industry_name"].nunique() if "industry_name" in const.columns else 0
        if industries < 20:
            const = _build_sw_constituents_from_sw_industry(tushare_root)
            const.to_parquet(const_path, index=False)

    hk = pd.read_parquet(hk_path)
    const = const.rename(columns={"con_code": "ts_code"})
    if "ts_code" not in const.columns:
        raise KeyError(f"sw_constituents 缺少 ts_code/con_code: {const.columns.tolist()}")
    const["ts_code"] = const["ts_code"].astype(str)
    const = const[const["ts_code"].map(_looks_like_a_share)]
    const["in_date"] = pd.to_datetime(const["in_date"])
    const["out_date"] = pd.to_datetime(const["out_date"])
    const["out_date"] = const["out_date"].fillna(pd.Timestamp("2099-12-31"))
    return hk, const


def _load_market_trade_dates(tushare_root: Path) -> pd.Series:
    candidates = [
        tushare_root / "northbound" / "northbound_daily_flow.parquet",
        tushare_root / "northbound" / "daily_flow.parquet",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            frame = pd.read_parquet(path, columns=["trade_date"])
        except Exception:
            continue
        if frame.empty:
            continue
        dates = pd.to_datetime(frame["trade_date"], errors="coerce").dropna().drop_duplicates().sort_values()
        if not dates.empty:
            return dates
    return pd.Series(dtype="datetime64[ns]")


def _extend_with_proxy_dates(agg: pd.DataFrame, market_dates: pd.Series) -> pd.DataFrame:
    if agg.empty or market_dates.empty:
        agg["is_proxy"] = False
        return agg

    result = agg.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"]).dt.floor("D")
    result["is_proxy"] = False

    last_ratio_date = result["trade_date"].max()
    target_last_date = market_dates.max()
    if pd.isna(last_ratio_date) or pd.isna(target_last_date) or target_last_date <= last_ratio_date:
        return result

    forward_dates = market_dates[market_dates > last_ratio_date]
    if forward_dates.empty:
        return result

    template = result[result["trade_date"] == last_ratio_date][
        ["industry_code", "industry_name", "industry_ratio"]
    ].copy()
    if template.empty:
        return result

    template["vol"] = pd.NA
    proxy_frames = []
    for trade_date in forward_dates.tolist():
        proxy = template.copy()
        proxy["trade_date"] = trade_date
        proxy["ratio"] = proxy["industry_ratio"]
        proxy["is_proxy"] = True
        proxy_frames.append(proxy)

    if not proxy_frames:
        return result

    result = pd.concat([result, *proxy_frames], ignore_index=True)
    result = result.sort_values(["trade_date", "industry_code"]).reset_index(drop=True)
    return result


def aggregate(tushare_root: Path) -> Path:
    hk, const = load_inputs(tushare_root)
    hk["trade_date"] = pd.to_datetime(hk["trade_date"])
    hk["ts_code"] = hk["ts_code"].astype(str)
    hk = hk[hk["ts_code"].map(_looks_like_a_share)].copy()
    hk["vol"] = pd.to_numeric(hk["vol"], errors="coerce")
    hk["ratio"] = pd.to_numeric(hk["ratio"], errors="coerce")

    merged = hk.merge(
        const[["ts_code", "industry_code", "industry_name", "in_date", "out_date"]],
        on="ts_code",
        how="left",
    )
    valid = merged[(merged["trade_date"] >= merged["in_date"]) & (merged["trade_date"] <= merged["out_date"])].copy()

    if valid.empty:
        raise RuntimeError("无法匹配行业成分（无有效记录）")

    agg = (
        valid.groupby(["industry_code", "industry_name", "trade_date"])
        .agg(
            vol=("vol", "sum"),
            ratio=("ratio", "mean"),
        )
        .reset_index()
    )

    total_vol = agg.groupby("trade_date")["vol"].transform("sum")
    agg["industry_ratio"] = agg["vol"] / total_vol.replace(0, pd.NA)
    market_dates = _load_market_trade_dates(tushare_root)
    agg = _extend_with_proxy_dates(agg, market_dates)

    output_path = tushare_root / "northbound" / "industry_northbound_flow.parquet"
    agg.to_parquet(output_path, index=False)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tushare-root", default=None)
    args = parser.parse_args()

    tushare_root = Path(args.tushare_root) if args.tushare_root else get_tushare_root()
    output_path = aggregate(tushare_root)
    output_df = pd.read_parquet(output_path)
    proxy_rows = int(pd.to_numeric(output_df.get("is_proxy", False), errors="coerce").fillna(False).astype(bool).sum())
    latest_date = pd.to_datetime(output_df["trade_date"], errors="coerce").max()
    latest_date_str = latest_date.strftime("%Y-%m-%d") if pd.notna(latest_date) else "unknown"
    print(f"北向行业配置已生成: {output_path}")
    print(f"  最新日期: {latest_date_str}, 代理行数: {proxy_rows}")


if __name__ == "__main__":
    main()
