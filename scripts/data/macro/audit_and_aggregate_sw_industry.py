#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
申万一级行业对齐审计 + 聚合特征生成

功能：
1) 基于 sw_nbs_mapping.yaml 的行业清单做覆盖率审计
2) 汇总 ppi/fai/output 等行业特征到统一表
3) 输出审计报告与聚合特征文件

用法:
  python scripts/data/macro/audit_and_aggregate_sw_industry.py \
    --tushare-root /Volumes/SPEED/BizData/Stock/Sage/data/tushare \
    --processed-dir /Volumes/SPEED/BizData/Stock/Sage/data/processed \
    --output-dir /Volumes/SPEED/BizData/Stock/Sage/data/processed
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import yaml
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data._shared.runtime import get_tushare_root, get_data_path


def load_mapping(config_path: Path) -> list[str]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return list((cfg.get("sw_to_nbs") or {}).keys())


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_ppi_sw(tushare_root: Path) -> pd.DataFrame:
    path = tushare_root / "macro" / "sw_l1_ppi_yoy_202512.csv"
    df = _safe_read_csv(path)
    if df.empty:
        return df
    df = df.rename(columns={"sw_ppi_yoy": "ppi_yoy"})
    df["date"] = pd.to_datetime(df["date"])
    return df[["sw_industry", "date", "ppi_yoy", "source_nbs_count"]]


def load_fai_sw(processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / "fai_sw_industry.parquet"
    df = _safe_read_parquet(path)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df[["sw_industry", "date", "fai_yoy", "fai_mom"]]


def load_output_sw(processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / "nbs_industrial_aligned.parquet"
    df = _safe_read_parquet(path)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    keep_cols = ["sw_industry", "date"]
    for col in ["output_yoy", "output_mom", "output_output_mom", "output_value"]:
        if col in df.columns:
            keep_cols.append(col)
    return df[keep_cols]


def audit_coverage(sw_list: list[str], datasets: dict) -> pd.DataFrame:
    rows = []
    missing_rows = []
    for name, df in datasets.items():
        sw_set = set(df["sw_industry"].dropna().unique()) if not df.empty else set()
        missing = [s for s in sw_list if s not in sw_set]
        last_date = df["date"].max() if ("date" in df.columns and not df.empty) else None
        rows.append(
            {
                "dataset": name,
                "industry_count": len(sw_set),
                "missing_count": len(missing),
                "coverage_ratio": round(len(sw_set) / len(sw_list), 4) if sw_list else 0,
                "last_date": last_date,
            }
        )
        for sw in missing:
            missing_rows.append({"dataset": name, "sw_industry": sw})

    summary = pd.DataFrame(rows)
    missing_df = pd.DataFrame(missing_rows)
    return summary, missing_df


def aggregate_features(ppi_df: pd.DataFrame, fai_df: pd.DataFrame, output_df: pd.DataFrame) -> pd.DataFrame:
    dfs = []
    if not ppi_df.empty:
        dfs.append(ppi_df)
    if not fai_df.empty:
        dfs.append(fai_df)
    if not output_df.empty:
        dfs.append(output_df)

    if not dfs:
        return pd.DataFrame()

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on=["sw_industry", "date"], how="outer")
    return merged.sort_values(["sw_industry", "date"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tushare-root", default=None, help="Tushare根目录（包含macro/）")
    parser.add_argument("--processed-dir", default=None, help="processed目录")
    parser.add_argument("--output-dir", default=None, help="输出目录")
    parser.add_argument("--config", default="config/sw_nbs_mapping.yaml")
    args = parser.parse_args()

    tushare_root = Path(args.tushare_root) if args.tushare_root else get_tushare_root()
    processed_dir = Path(args.processed_dir) if args.processed_dir else get_data_path("processed")
    output_dir = Path(args.output_dir) if args.output_dir else processed_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sw_list = load_mapping(PROJECT_ROOT / args.config)

    ppi_df = load_ppi_sw(tushare_root)
    fai_df = load_fai_sw(processed_dir)
    output_df = load_output_sw(processed_dir)

    summary, missing = audit_coverage(
        sw_list,
        {"ppi_yoy": ppi_df, "fai_yoy": fai_df, "output": output_df},
    )

    timestamp = datetime.now().strftime("%Y%m%d")
    summary_path = output_dir / f"macro_industry_audit_summary_{timestamp}.parquet"
    missing_path = output_dir / f"macro_industry_missing_{timestamp}.parquet"
    summary.to_parquet(summary_path, index=False)
    missing.to_parquet(missing_path, index=False)

    features = aggregate_features(ppi_df, fai_df, output_df)
    features_path = output_dir / "industry_features_sw_l1.parquet"
    if not features.empty:
        features.to_parquet(features_path, index=False)

    print("对齐审计完成:")
    print(summary)
    print(f"审计汇总: {summary_path}")
    print(f"缺失清单: {missing_path}")
    if not features.empty:
        print(f"行业聚合特征: {features_path}")
    else:
        print("行业聚合特征: 无数据")


if __name__ == "__main__":
    main()
