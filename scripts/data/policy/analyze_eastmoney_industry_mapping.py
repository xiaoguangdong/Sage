#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
东方财富行业研报 → 申万L1 行业映射覆盖率统计
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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


def map_one(name: str, l2_to_l1: Dict[str, str], l1_set: set, alias_map: Dict[str, str]) -> str:
    if not name:
        return ""
    alias = alias_map.get(name)
    if alias:
        if alias in l1_set:
            return alias
        if alias in l2_to_l1:
            return l2_to_l1[alias]
    if name in l1_set:
        return name
    if name in l2_to_l1:
        return l2_to_l1[name]
    normalized = _normalize_industry_name(name)
    if normalized in l1_set:
        return normalized
    if normalized in l2_to_l1:
        return l2_to_l1[normalized]
    return ""


def main() -> None:
    input_path = get_data_path("raw", "policy") / "eastmoney_industry_reports.parquet"
    if not input_path.exists():
        print(f"未找到输入文件: {input_path}")
        return

    df = pd.read_parquet(input_path)
    if "industry" not in df.columns:
        print("缺少 industry 字段，无法统计")
        return

    l2_to_l1 = load_sw_l2_to_l1_map()
    l1_set = set(l2_to_l1.values())
    alias_cfg = load_yaml(PROJECT_ROOT / "config" / "policy_industry_alias.yaml")
    alias_map = (alias_cfg.get("aliases") or {}) if isinstance(alias_cfg, dict) else {}

    counts = df["industry"].fillna("").astype(str).value_counts()
    total_records = int(len(df))
    total_unique = int(len(counts))

    mapped_records = 0
    mapped_unique = 0
    mapped_by_l1: Dict[str, int] = {}
    unmatched_rows: List[Tuple[str, int]] = []

    for name, cnt in counts.items():
        mapped = map_one(name, l2_to_l1, l1_set, alias_map)
        if mapped:
            mapped_unique += 1
            mapped_records += int(cnt)
            mapped_by_l1[mapped] = mapped_by_l1.get(mapped, 0) + int(cnt)
        else:
            unmatched_rows.append((name, int(cnt)))

    output_dir = get_data_path("processed", "policy", ensure=True)
    report = {
        "input": str(input_path),
        "total_records": total_records,
        "total_unique_industries": total_unique,
        "mapped_records": mapped_records,
        "mapped_unique_industries": mapped_unique,
        "coverage_records": round(mapped_records / total_records, 4) if total_records else 0,
        "coverage_unique": round(mapped_unique / total_unique, 4) if total_unique else 0,
        "top_l1_by_records": sorted(mapped_by_l1.items(), key=lambda x: x[1], reverse=True)[:20],
        "unmatched_count": len(unmatched_rows),
    }

    (output_dir / "eastmoney_industry_mapping_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if unmatched_rows:
        pd.DataFrame(unmatched_rows, columns=["industry", "records"]).to_csv(
            output_dir / "eastmoney_industry_unmatched.csv", index=False
        )

    print(f"映射报告已保存: {output_dir / 'eastmoney_industry_mapping_report.json'}")
    if unmatched_rows:
        print(f"未映射清单: {output_dir / 'eastmoney_industry_unmatched.csv'}")


if __name__ == "__main__":
    main()
