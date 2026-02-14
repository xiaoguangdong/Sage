#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于概念成分股生成“概念→申万L1行业”映射与覆盖率统计
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from scripts.data._shared.runtime import get_data_path


def _pick_col(columns: List[str], candidates: List[str]) -> Optional[str]:
    for name in candidates:
        if name in columns:
            return name
    return None


def load_concept_detail(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"概念成分文件不存在: {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return df


def load_sw_l1(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"申万L1分类不存在: {path}")
    return pd.read_parquet(path)


def load_sw_index_member(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"申万成分股不存在: {path}")
    return pd.read_parquet(path)


def build_stock_industry_map(members: pd.DataFrame, l1: pd.DataFrame) -> pd.DataFrame:
    name_col = "industry_name" if "industry_name" in l1.columns else "index_name"
    l1 = l1.rename(columns={name_col: "industry_name"})
    members = members.rename(columns={"con_code": "ts_code"})
    merged = members.merge(l1[["index_code", "industry_name"]], on="index_code", how="left")
    merged = merged.dropna(subset=["ts_code", "industry_name"])
    return merged[["ts_code", "industry_name"]].drop_duplicates()


def build_mapping(
    concept_df: pd.DataFrame,
    stock_map: pd.DataFrame,
    min_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    cols = concept_df.columns.tolist()
    concept_code_col = _pick_col(cols, ["id", "concept_id", "concept_code", "code"])
    concept_name_col = _pick_col(cols, ["concept_name", "name", "concept"])
    stock_col = _pick_col(cols, ["ts_code", "con_code", "stock_code", "symbol"])

    if not stock_col:
        raise ValueError("概念成分缺少股票列（ts_code/con_code/symbol）")

    df = concept_df.copy()
    df = df.rename(columns={stock_col: "ts_code"})
    if concept_code_col:
        df = df.rename(columns={concept_code_col: "concept_code"})
    if concept_name_col:
        df = df.rename(columns={concept_name_col: "concept_name"})

    if "concept_code" not in df.columns:
        df["concept_code"] = df["ts_code"].astype(str)
    if "concept_name" not in df.columns:
        df["concept_name"] = df["concept_code"]

    df["ts_code"] = df["ts_code"].astype(str)

    merged = df.merge(stock_map, on="ts_code", how="left")
    total_by_concept = merged.groupby("concept_code")["ts_code"].nunique().rename("total_stocks")
    mapped = merged.dropna(subset=["industry_name"]).copy()

    if mapped.empty:
        coverage = pd.DataFrame(columns=["concept_code", "concept_name", "sw_industry", "stock_count", "stock_ratio"])
        primary = pd.DataFrame(columns=["concept_code", "concept_name", "sw_industry", "stock_ratio", "total_stocks"])
        unmapped = total_by_concept.reset_index()
        report = {
            "total_concepts": int(total_by_concept.shape[0]),
            "mapped_concepts": 0,
            "coverage_rate": 0,
        }
        return coverage, primary, unmapped, report

    mapped["stock_count"] = 1
    grouped = mapped.groupby(["concept_code", "concept_name", "industry_name"]).agg(
        stock_count=("stock_count", "sum"),
        stock_count_unique=("ts_code", "nunique"),
    ).reset_index()
    grouped["stock_count"] = grouped["stock_count_unique"]

    grouped = grouped.drop(columns=["stock_count_unique"])
    grouped = grouped.rename(columns={"industry_name": "sw_industry"})
    grouped = grouped.merge(total_by_concept.reset_index(), on="concept_code", how="left")
    grouped["stock_ratio"] = grouped["stock_count"] / grouped["total_stocks"].replace(0, pd.NA)

    coverage = grouped.sort_values(["concept_code", "stock_ratio"], ascending=[True, False])

    primary = coverage.groupby("concept_code").head(1).copy()
    primary = primary[primary["stock_ratio"] >= min_ratio]
    primary = primary[["concept_code", "concept_name", "sw_industry", "stock_ratio", "total_stocks"]]

    mapped_concepts = set(primary["concept_code"].unique().tolist())
    all_concepts = set(total_by_concept.index.tolist())
    unmapped_concepts = sorted(all_concepts - mapped_concepts)
    unmapped = total_by_concept.reset_index()
    unmapped = unmapped[unmapped["concept_code"].isin(unmapped_concepts)]

    report = {
        "total_concepts": int(len(all_concepts)),
        "mapped_concepts": int(len(mapped_concepts)),
        "coverage_rate": round(len(mapped_concepts) / len(all_concepts), 4) if all_concepts else 0,
        "min_ratio": min_ratio,
    }

    return coverage, primary, unmapped, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-ratio", type=float, default=0.2, help="主行业最小覆盖率阈值")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else get_data_path("processed", "concepts", ensure=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    concept_path = get_data_path("raw", "tushare", "sectors") / "concept_detail.parquet"
    l1_path = get_data_path("raw", "tushare", "sw_industry") / "sw_industry_l1.parquet"
    member_path = get_data_path("raw", "tushare", "sw_industry") / "sw_index_member.parquet"

    concept_df = load_concept_detail(concept_path)
    l1_df = load_sw_l1(l1_path)
    member_df = load_sw_index_member(member_path)
    stock_map = build_stock_industry_map(member_df, l1_df)

    coverage, primary, unmapped, report = build_mapping(concept_df, stock_map, min_ratio=args.min_ratio)

    coverage.to_parquet(output_dir / "concept_industry_coverage.parquet", index=False)
    primary.to_parquet(output_dir / "concept_industry_primary.parquet", index=False)
    unmapped.to_parquet(output_dir / "concept_industry_unmapped.parquet", index=False)
    (output_dir / "concept_industry_mapping_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"覆盖率表: {output_dir / 'concept_industry_coverage.parquet'}")
    print(f"主行业表: {output_dir / 'concept_industry_primary.parquet'}")
    print(f"未映射概念: {output_dir / 'concept_industry_unmapped.parquet'}")


if __name__ == "__main__":
    main()
