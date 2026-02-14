#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
政策行业信号增强：融合行业研报 + 申万行业指数动量
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data._shared.runtime import get_data_path, get_data_root


def load_yaml(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _normalize_industry_name(name: str) -> str:
    return re.sub(r"[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+$", "", name or "").strip()


def load_sw_l2_to_l1_map() -> Dict[str, str]:
    data_root = get_data_root()
    l2_path = data_root / "tushare" / "sectors" / "SW2021_L2_classify.csv"
    l1_path = data_root / "tushare" / "sectors" / "SW2021_L1_classify.csv"
    if not l2_path.exists() or not l1_path.exists():
        return {}
    l2 = pd.read_csv(l2_path)
    l1 = pd.read_csv(l1_path)
    l1_map = dict(zip(l1["industry_code"].astype(str), l1["industry_name"].astype(str)))
    mapping: Dict[str, str] = {}
    for _, row in l2.iterrows():
        name = str(row.get("industry_name", "")).strip()
        parent = str(row.get("parent_code", "")).strip()
        if not name or not parent:
            continue
        l1_name = l1_map.get(parent)
        if l1_name:
            mapping[name] = l1_name
    return mapping


def load_sw_l1_map() -> Dict[str, str]:
    data_root = get_data_root()
    l1_path = data_root / "tushare" / "sectors" / "SW2021_L1_classify.csv"
    if not l1_path.exists():
        return {}
    l1 = pd.read_csv(l1_path)
    return dict(zip(l1["index_code"].astype(str), l1["industry_name"].astype(str)))


def map_industry_name(raw: str, l2_to_l1: Dict[str, str], l1_set: set, alias_map: Dict[str, str]) -> List[str]:
    if not raw:
        return []
    parts = re.split(r"[、,;/|]", raw)
    results: List[str] = []
    for part in parts:
        name = part.strip()
        if not name:
            continue
        alias = alias_map.get(name)
        if alias:
            if alias in l1_set:
                results.append(alias)
                continue
            if alias in l2_to_l1:
                results.append(l2_to_l1[alias])
                continue
        if name in l1_set:
            results.append(name)
            continue
        if name in l2_to_l1:
            results.append(l2_to_l1[name])
            continue
        normalized = _normalize_industry_name(name)
        if normalized in l1_set:
            results.append(normalized)
            continue
        if normalized in l2_to_l1:
            results.append(l2_to_l1[normalized])
            continue
    return list(dict.fromkeys(results))


def build_report_factors(reports_path: Path, alias_map: Dict[str, str]) -> pd.DataFrame:
    if not reports_path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(reports_path)
    if df.empty or "industry" not in df.columns:
        return pd.DataFrame()

    l2_to_l1 = load_sw_l2_to_l1_map()
    l1_set = set(l2_to_l1.values())

    df = df.copy()
    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
    df = df.dropna(subset=["publish_date"])
    mapped = []
    for _, row in df.iterrows():
        industries = map_industry_name(str(row.get("industry", "")), l2_to_l1, l1_set, alias_map)
        if not industries:
            continue
        for ind in industries:
            mapped.append({
                "trade_date": row["publish_date"].normalize(),
                "sw_industry": ind,
                "rating_change": str(row.get("rating_change", "")).strip(),
            })
    if not mapped:
        return pd.DataFrame()

    mdf = pd.DataFrame(mapped)
    mdf["trade_date"] = pd.to_datetime(mdf["trade_date"])
    mdf["rc_score"] = mdf["rating_change"].map({
        "上调": 1,
        "下调": -1,
        "维持": 0,
        "首次": 0,
        "调高": 1,
        "调低": -1,
    }).fillna(0)

    factors = []
    for ind, group in mdf.groupby("sw_industry"):
        g = group.set_index("trade_date").sort_index()
        daily_cnt = g["rc_score"].resample("D").size().rename("report_count")
        daily_score = g["rc_score"].resample("D").sum().rename("rating_change_score")
        frame = pd.concat([daily_cnt, daily_score], axis=1).fillna(0)
        frame["report_count_7d"] = frame["report_count"].rolling("7D").sum()
        frame["report_count_30d"] = frame["report_count"].rolling("30D").sum()
        up = g["rc_score"].resample("D").apply(lambda x: (x > 0).sum()).fillna(0)
        down = g["rc_score"].resample("D").apply(lambda x: (x < 0).sum()).fillna(0)
        frame["rating_up_30d"] = up.rolling("30D").sum()
        frame["rating_down_30d"] = down.rolling("30D").sum()
        frame["rating_change_30d"] = frame["rating_up_30d"] - frame["rating_down_30d"]
        frame["sw_industry"] = ind
        frame = frame.reset_index().rename(columns={"index": "trade_date"})
        factors.append(frame)

    if not factors:
        return pd.DataFrame()
    return pd.concat(factors, ignore_index=True)


def build_industry_momentum(sw_daily_path: Path) -> pd.DataFrame:
    if not sw_daily_path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(sw_daily_path)
    if df.empty:
        return pd.DataFrame()
    l1_map = load_sw_l1_map()
    df = df.copy()
    df["sw_industry"] = df["ts_code"].astype(str).map(l1_map)
    df = df.dropna(subset=["sw_industry"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["sw_industry", "trade_date"])

    df["mom_20d"] = df.groupby("sw_industry")["close"].transform(lambda s: s / s.shift(20) - 1)
    df["mom_60d"] = df.groupby("sw_industry")["close"].transform(lambda s: s / s.shift(60) - 1)
    df["vol_20d"] = df.groupby("sw_industry")["pct_change"].transform(lambda s: s.rolling(20).std())
    return df[["trade_date", "sw_industry", "mom_20d", "mom_60d", "vol_20d"]]


def merge_asof_by_industry(left: pd.DataFrame, right: pd.DataFrame, on: str) -> pd.DataFrame:
    if right.empty:
        return left
    left = left.copy()
    right = right.copy()
    left[on] = pd.to_datetime(left[on]).dt.floor("D").astype("datetime64[ns]")
    right[on] = pd.to_datetime(right[on]).dt.floor("D").astype("datetime64[ns]")
    left = left.sort_values(["sw_industry", on])
    right = right.sort_values(["sw_industry", on])
    result = []
    for ind, group in left.groupby("sw_industry"):
        r = right[right["sw_industry"] == ind]
        if r.empty:
            result.append(group)
            continue
        merged = pd.merge_asof(
            group.sort_values(on),
            r.sort_values(on),
            on=on,
            by="sw_industry",
            direction="backward",
        )
        result.append(merged)
    return pd.concat(result, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-signals", type=str, default=None)
    parser.add_argument("--reports", type=str, default=None)
    parser.add_argument("--sw-daily", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    policy_signals = Path(args.policy_signals) if args.policy_signals else get_data_path(
        "processed", "policy", "policy_signals.parquet"
    )
    reports_path = Path(args.reports) if args.reports else get_data_path(
        "raw", "policy", "eastmoney_industry_reports.parquet"
    )
    sw_daily_path = Path(args.sw_daily) if args.sw_daily else get_data_root() / "tushare" / "sectors" / "sw_daily_all.parquet"
    output_dir = Path(args.output_dir) if args.output_dir else get_data_path("processed", "policy", ensure=True)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not policy_signals.exists():
        print(f"未找到政策信号: {policy_signals}")
        return

    signals = pd.read_parquet(policy_signals)
    signals["trade_date"] = pd.to_datetime(signals["trade_date"])

    alias_cfg = load_yaml(PROJECT_ROOT / "config" / "policy_industry_alias.yaml")
    alias_map = (alias_cfg.get("aliases") or {}) if isinstance(alias_cfg, dict) else {}

    report_factors = build_report_factors(reports_path, alias_map)
    momentum = build_industry_momentum(sw_daily_path)

    enhanced = signals.copy()
    if not report_factors.empty:
        enhanced = merge_asof_by_industry(enhanced, report_factors, "trade_date")
    if not momentum.empty:
        enhanced = merge_asof_by_industry(enhanced, momentum, "trade_date")

    output_path = output_dir / "policy_signals_enhanced.parquet"
    enhanced.to_parquet(output_path, index=False)

    summary = {
        "policy_signals": str(policy_signals),
        "reports_used": bool(not report_factors.empty),
        "sw_daily_used": bool(not momentum.empty),
        "rows": int(len(enhanced)),
    }
    (output_dir / "policy_signals_enhanced_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"增强信号已保存: {output_path}")


if __name__ == "__main__":
    main()
