#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
行业特征缺口诊断：
1) 统计缺失字段比例
2) 列出缺失最严重的行业/月份
3) 输出诊断报告（JSON + CSV）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data._shared.runtime import get_data_path
from scripts.data.macro.clean_macro_data import MacroDataProcessor


DEFAULT_FIELDS = ["rev_yoy", "sw_ppi_yoy", "inventory_yoy"]


def _ensure_processed_exists(processed_dir: Path) -> Path:
    industry_path = processed_dir / "industry_features.parquet"
    if industry_path.exists():
        return industry_path

    processor = MacroDataProcessor()
    data = processor.process_all()
    processed_dir.mkdir(parents=True, exist_ok=True)
    data["industry"].to_parquet(industry_path, index=False)
    if "macro" in data:
        data["macro"].to_parquet(processed_dir / "macro_features.parquet", index=False)
    if "northbound" in data and len(data["northbound"]) > 0:
        data["northbound"].to_parquet(processed_dir / "northbound_features.parquet", index=False)
    return industry_path


def diagnose(fields: List[str] | None = None) -> Dict:
    fields = fields or DEFAULT_FIELDS
    processed_dir = get_data_path("processed", ensure=True)
    industry_path = _ensure_processed_exists(processed_dir)

    df = pd.read_parquet(industry_path)
    if df.empty:
        return {"error": "industry_features.parquet为空"}

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    results = {}
    for col in fields:
        if col not in df.columns:
            results[col] = {"missing": "column_not_found"}
            continue

        missing_mask = df[col].isna()
        by_industry = (
            df.assign(missing=missing_mask)
            .groupby("sw_industry")["missing"]
            .mean()
            .sort_values(ascending=False)
        )
        by_date = (
            df.assign(missing=missing_mask)
            .groupby("date")["missing"]
            .mean()
            .sort_values(ascending=False)
        )

        results[col] = {
            "missing_ratio": float(missing_mask.mean()),
            "missing_by_industry": by_industry.head(30).round(6).to_dict(),
            "missing_by_date": {k.strftime("%Y-%m-%d"): float(v) for k, v in by_date.head(30).items()},
        }

    report_path = processed_dir / "industry_gap_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 输出一份扁平化 CSV 便于筛选
    flat_rows = []
    for col, detail in results.items():
        if "missing_ratio" not in detail:
            continue
        for industry, ratio in detail["missing_by_industry"].items():
            flat_rows.append({"field": col, "dimension": "industry", "key": industry, "missing_ratio": ratio})
        for date_str, ratio in detail["missing_by_date"].items():
            flat_rows.append({"field": col, "dimension": "date", "key": date_str, "missing_ratio": ratio})
    if flat_rows:
        pd.DataFrame(flat_rows).to_csv(processed_dir / "industry_gap_report.csv", index=False)

    return {"report": str(report_path), "rows": len(df), "fields": fields}


def main():
    result = diagnose()
    if "error" in result:
        print(result["error"])
    else:
        print(f"诊断完成: {result['report']}")
        print(f"样本行数: {result['rows']} 字段: {', '.join(result['fields'])}")


if __name__ == "__main__":
    main()
