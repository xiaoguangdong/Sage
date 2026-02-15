#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于同花顺概念成分(ths_member)生成“概念→申万L1行业”映射与覆盖率统计
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from scripts.data._shared.runtime import get_data_path

def load_ths_member(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"同花顺概念成分文件不存在: {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return df


def load_ths_index(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["ts_code", "name"])
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return df


def _looks_like_concept_code(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(50)
    if sample.empty:
        return False
    return sample.str.endswith(".TI").mean() > 0.6


def normalize_ths_member(member_df: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    cols = member_df.columns.tolist()
    concept_code_col = None
    stock_col = None
    concept_name_col = "concept_name" if "concept_name" in cols else None

    for candidate in ["concept_code", "index_code", "id", "ts_code", "code"]:
        if candidate in cols and _looks_like_concept_code(member_df[candidate]):
            concept_code_col = candidate
            break
    if concept_code_col is None:
        for candidate in ["concept_code", "index_code", "id", "ts_code", "code"]:
            if candidate in cols:
                concept_code_col = candidate
                break

    for candidate in ["ts_code", "con_code", "stock_code", "code", "symbol"]:
        if candidate in cols and candidate != concept_code_col:
            stock_col = candidate
            break

    if concept_code_col is None or stock_col is None:
        raise ValueError(f"ths_member 字段不足，无法识别概念与成分列: {cols}")

    normalized = member_df.rename(columns={concept_code_col: "concept_code", stock_col: "ts_code"}).copy()
    normalized["concept_code"] = normalized["concept_code"].astype(str)
    normalized["ts_code"] = normalized["ts_code"].astype(str)

    if not concept_name_col:
        for candidate in ["name", "index_name"]:
            if candidate in cols:
                concept_name_col = candidate
                break
    if concept_name_col:
        normalized = normalized.rename(columns={concept_name_col: "concept_name"})

    if "concept_name" not in normalized.columns and not index_df.empty and "ts_code" in index_df.columns:
        name_col = "name" if "name" in index_df.columns else ("index_name" if "index_name" in index_df.columns else None)
        if name_col:
            names = index_df[["ts_code", name_col]].rename(columns={"ts_code": "concept_code", name_col: "concept_name"})
            normalized = normalized.merge(names, on="concept_code", how="left")

    if "concept_name" not in normalized.columns:
        normalized["concept_name"] = normalized["concept_code"]

    return normalized[["concept_code", "concept_name", "ts_code"]].drop_duplicates()


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
    merged = concept_df.merge(stock_map, on="ts_code", how="left")
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

    concept_path = get_data_path("raw", "tushare", "concepts") / "ths_member.parquet"
    ths_index_path = get_data_path("raw", "tushare", "concepts") / "ths_index.parquet"
    l1_path = get_data_path("raw", "tushare", "sw_industry") / "sw_industry_l1.parquet"
    member_path = get_data_path("raw", "tushare", "sw_industry") / "sw_index_member.parquet"

    ths_member_df = load_ths_member(concept_path)
    ths_index_df = load_ths_index(ths_index_path)
    concept_df = normalize_ths_member(ths_member_df, ths_index_df)
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
