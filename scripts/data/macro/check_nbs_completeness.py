#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NBS 数据完整性检查（按月）
默认检查 2016-01 至当前月份，输出到 logs/data/
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class DatasetSpec:
    name: str
    files: List[str]
    kind: str  # "wide_month_columns" | "date_col" | "date_col_or_quarter"


DATASETS: List[DatasetSpec] = [
    DatasetSpec(
        name="nbs_cpi_national",
        files=["macro/nbs_cpi_national.csv"],
        kind="wide_month_columns",
    ),
    DatasetSpec(
        name="nbs_ppi_national",
        files=["macro/nbs_ppi_national.csv"],
        kind="wide_month_columns",
    ),
    DatasetSpec(
        name="nbs_ppi_industry",
        files=["macro/nbs_ppi_industry_2020.csv", "macro/nbs_ppi_industry_202512.csv"],
        kind="date_col",
    ),
    DatasetSpec(
        name="nbs_fai_industry",
        files=["macro/nbs_fai_industry_2020.csv", "macro/nbs_fai_industry_202512.csv"],
        kind="date_col_or_quarter",
    ),
    DatasetSpec(
        name="nbs_output",
        files=["macro/nbs_output_2020.csv", "macro/nbs_output_202512.csv"],
        kind="date_col",
    ),
]


def _load_base_paths() -> Tuple[Path, Path]:
    base_yaml = PROJECT_ROOT / "config" / "base.yaml"
    if base_yaml.exists():
        cfg = yaml.safe_load(base_yaml.read_text(encoding="utf-8")) or {}
        primary = Path((cfg.get("data") or {}).get("roots", {}).get("primary", PROJECT_ROOT / "data"))
        tushare_cfg = (cfg.get("data") or {}).get("paths", {}).get("tushare", "data/tushare")
        tushare_path = Path(tushare_cfg)
        if not tushare_path.is_absolute():
            tushare_path = PROJECT_ROOT / tushare_path
        return primary, tushare_path
    return PROJECT_ROOT / "data", PROJECT_ROOT / "data" / "tushare"


def _resolve_files(files: List[str]) -> List[Path]:
    primary_root, tushare_path = _load_base_paths()
    candidates: List[Path] = []
    for rel in files:
        p1 = tushare_path / rel
        p2 = primary_root / "raw" / "tushare" / rel
        if p1.exists():
            candidates.append(p1)
        elif p2.exists():
            candidates.append(p2)
    return candidates


def _parse_month_columns(columns: List[str]) -> List[pd.Period]:
    months: List[pd.Period] = []
    for col in columns:
        name = str(col).strip()
        m = re.match(r"^(\d{4})年(\d{1,2})月$", name)
        if not m:
            continue
        y, mo = int(m.group(1)), int(m.group(2))
        months.append(pd.Period(f"{y}-{mo:02d}", freq="M"))
    return months


def _parse_quarter_to_month(value: str) -> Optional[pd.Period]:
    if not isinstance(value, str):
        return None
    m = re.search(r"(\d{4})-?Q([1-4])", value)
    if m:
        year = int(m.group(1))
        q = int(m.group(2))
        month = q * 3
        return pd.Period(f"{year}-{month:02d}", freq="M")
    m = re.search(r"(\d{4})-?Q第?([一二三四])", value)
    if m:
        year = int(m.group(1))
        q_map = {"一": 1, "二": 2, "三": 3, "四": 4}
        month = q_map[m.group(2)] * 3
        return pd.Period(f"{year}-{month:02d}", freq="M")
    return None


def _collect_months_from_date_column(df: pd.DataFrame, allow_quarter: bool = False) -> Tuple[List[pd.Period], int]:
    if "date" not in df.columns:
        return [], 0
    series = df["date"].dropna().astype(str)
    dates = pd.to_datetime(series, errors="coerce")
    months = [pd.Period(d.strftime("%Y-%m"), freq="M") for d in dates.dropna().unique()]
    unparsed = int(series.size - dates.notna().sum())
    if allow_quarter and unparsed > 0:
        for val in series[dates.isna()].unique():
            period = _parse_quarter_to_month(val)
            if period:
                months.append(period)
                unparsed -= 1
    return months, max(unparsed, 0)


def _expected_months(start: str, end: str) -> List[pd.Period]:
    start_p = pd.Period(start, freq="M")
    end_p = pd.Period(end, freq="M")
    return list(pd.period_range(start=start_p, end=end_p, freq="M"))


def _render_month_list(months: List[pd.Period], limit: int = 12) -> List[str]:
    return [str(m) for m in months[:limit]]


def check_dataset(spec: DatasetSpec, start: str, end: str) -> Dict:
    files = _resolve_files(spec.files)
    if not files:
        return {
            "dataset": spec.name,
            "status": "missing",
            "files": [],
        }

    all_months: List[pd.Period] = []
    unparsed_total = 0
    for path in files:
        df = pd.read_csv(path)
        if spec.kind == "wide_month_columns":
            months = _parse_month_columns(list(df.columns))
            all_months.extend(months)
        elif spec.kind == "date_col":
            months, unparsed = _collect_months_from_date_column(df, allow_quarter=False)
            all_months.extend(months)
            unparsed_total += unparsed
        elif spec.kind == "date_col_or_quarter":
            months, unparsed = _collect_months_from_date_column(df, allow_quarter=True)
            all_months.extend(months)
            unparsed_total += unparsed

    unique_months = sorted(set(all_months))
    expected = _expected_months(start, end)
    missing = sorted(set(expected) - set(unique_months))
    return {
        "dataset": spec.name,
        "status": "ok",
        "files": [str(p) for p in files],
        "min_month": str(unique_months[0]) if unique_months else None,
        "max_month": str(unique_months[-1]) if unique_months else None,
        "expected_months": len(expected),
        "present_months": len(unique_months),
        "missing_months": [str(m) for m in missing],
        "missing_months_head": _render_month_list(missing),
        "unparsed_count": unparsed_total,
    }


def main() -> int:
    today = date.today()
    default_end = f"{today.year}-{today.month:02d}"
    parser = argparse.ArgumentParser(description="NBS 数据完整性检查")
    parser.add_argument("--start", default="2016-01", help="起始月份 YYYY-MM")
    parser.add_argument("--end", default=default_end, help="结束月份 YYYY-MM")
    parser.add_argument("--output-dir", default="logs/data", help="报告输出目录")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = today.strftime("%Y%m%d")
    report_path = out_dir / f"nbs_completeness_report_{stamp}.txt"
    json_path = out_dir / f"nbs_completeness_report_{stamp}.json"

    results = [check_dataset(spec, args.start, args.end) for spec in DATASETS]
    summary = {
        "start": args.start,
        "end": args.end,
        "datasets": len(results),
        "missing_datasets": [r["dataset"] for r in results if r["status"] != "ok"],
    }
    payload = {"summary": summary, "results": results}

    lines = [
        "NBS 数据完整性检查",
        f"时间范围: {args.start} ~ {args.end}",
        "========================================",
    ]
    for r in results:
        lines.append(f"\n[{r['dataset']}]")
        if r["status"] != "ok":
            lines.append("  ❌ 文件缺失")
            continue
        lines.append(f"  文件: {', '.join(r['files'])}")
        lines.append(f"  覆盖: {r['min_month']} ~ {r['max_month']}")
        lines.append(f"  覆盖月份: {r['present_months']}/{r['expected_months']}")
        if r["missing_months"]:
            lines.append(f"  缺失月份(前12): {', '.join(r['missing_months_head'])}")
        if r["unparsed_count"]:
            lines.append(f"  ⚠️ 未解析日期数量: {r['unparsed_count']}")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已保存: {report_path}")
    print(f"JSON已保存: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
