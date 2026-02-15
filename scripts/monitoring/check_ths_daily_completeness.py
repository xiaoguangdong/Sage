#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from scripts.data._shared.runtime import get_data_path, get_tushare_root


def _locate_ths_daily() -> Path:
    candidates = [
        get_tushare_root() / "concepts" / "ths_daily.parquet",
        get_data_path("raw", "tushare", "concepts", "ths_daily.parquet"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"未找到 ths_daily，候选路径: {candidates}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ths_daily 按月完整性检查")
    parser.add_argument("--max-stale-days", type=int, default=7)
    parser.add_argument("--min-monthly-ratio", type=float, default=0.8, help="月数据量低于近12月中位数的比例阈值")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    path = _locate_ths_daily()
    df = pd.read_parquet(path, columns=["ts_code", "trade_date"])
    if df.empty:
        raise SystemExit("ths_daily 为空")

    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df.dropna(subset=["trade_date"])
    df["month"] = df["trade_date"].dt.to_period("M").astype(str)

    monthly = (
        df.groupby("month", as_index=False)
        .agg(
            rows=("ts_code", "size"),
            concept_count=("ts_code", "nunique"),
            trade_days=("trade_date", "nunique"),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )

    period_index = pd.period_range(start=monthly["month"].iloc[0], end=monthly["month"].iloc[-1], freq="M")
    expected_months = set(period_index.astype(str))
    existing_months = set(monthly["month"].tolist())
    missing_months = sorted(expected_months - existing_months)

    latest_trade_date = df["trade_date"].max().normalize()
    stale_days = int((pd.Timestamp(datetime.now().date()) - latest_trade_date).days)

    ref = monthly.tail(12)["rows"]
    median_rows = float(ref.median()) if not ref.empty else float(monthly["rows"].median())
    monthly["row_ratio"] = monthly["rows"] / median_rows if median_rows > 0 else 1.0
    low_months = monthly[monthly["row_ratio"] < float(args.min_monthly_ratio)][["month", "rows", "row_ratio"]]

    checks = {
        "latest_data_ok": stale_days <= int(args.max_stale_days),
        "missing_months_ok": len(missing_months) == 0,
        "monthly_volume_ok": len(low_months) == 0,
    }
    payload = {
        "passed": all(checks.values()),
        "source_path": str(path),
        "checks": checks,
        "metrics": {
            "latest_trade_date": latest_trade_date.strftime("%Y-%m-%d"),
            "stale_days": stale_days,
            "months_total": int(len(monthly)),
            "rows_total": int(len(df)),
            "concept_count": int(df["ts_code"].nunique()),
            "median_monthly_rows_recent12": median_rows,
        },
        "issues": {
            "missing_months": missing_months,
            "low_months": low_months.to_dict(orient="records"),
        },
        "thresholds": {
            "max_stale_days": int(args.max_stale_days),
            "min_monthly_ratio": float(args.min_monthly_ratio),
        },
    }

    out = Path(args.output) if args.output else get_data_path("processed", "concepts", "ths_daily_completeness_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完整性报告已保存: {out}")
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
