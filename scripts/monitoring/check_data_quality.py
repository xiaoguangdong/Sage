#!/usr/bin/env python3
import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd


SUPPORTED_EXTS = {".csv", ".parquet"}
DATE_COLUMNS = {"trade_date", "date", "month", "ann_date", "end_date", "cal_date"}
KEY_CANDIDATES = [
    ("ts_code", "trade_date"),
    ("code", "date"),
    ("stock", "date"),
    ("ts_code", "date"),
]


def list_data_files(root: Path) -> list[Path]:
    files = []
    for ext in SUPPORTED_EXTS:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)


def _coerce_datetime(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    sample = series.dropna().astype(str).head(20)
    if sample.empty:
        return pd.to_datetime(series, errors="coerce")
    sample_str = sample.iloc[0]
    if len(sample_str) == 8 and sample_str.isdigit():
        return pd.to_datetime(series.astype(str), format="%Y%m%d", errors="coerce")
    if len(sample_str) == 6 and sample_str.isdigit():
        return pd.to_datetime(series.astype(str), format="%Y%m", errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def summarize_df(df: pd.DataFrame) -> dict:
    n_rows, n_cols = df.shape
    missing_total = int(df.isna().sum().sum())
    total_cells = n_rows * n_cols
    missing_ratio = float(missing_total / total_cells) if total_cells else 0.0

    date_summary = {}
    for col in df.columns:
        if col in DATE_COLUMNS or col.endswith("_date") or "date" == col:
            parsed = _coerce_datetime(df[col])
            if parsed.notna().any():
                date_summary[col] = {
                    "min": str(parsed.min().date()),
                    "max": str(parsed.max().date()),
                    "missing": int(parsed.isna().sum()),
                }

    duplicate_summary = {}
    for keys in KEY_CANDIDATES:
        if all(k in df.columns for k in keys):
            duplicate_summary["+".join(keys)] = int(df.duplicated(subset=list(keys)).sum())

    missing_by_col = (
        df.isna().mean().sort_values(ascending=False).head(10).to_dict()
        if n_cols > 0
        else {}
    )

    return {
        "rows": n_rows,
        "cols": n_cols,
        "missing_total": missing_total,
        "missing_ratio": round(missing_ratio, 6),
        "missing_top10": {k: round(v, 6) for k, v in missing_by_col.items()},
        "dates": date_summary,
        "duplicates": duplicate_summary,
    }


def read_file(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix == ".csv":
            return pd.read_csv(path)
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
    except Exception:
        return None
    return None


def render_report(summary: dict, output_path: Path) -> None:
    lines = []
    lines.append("# 数据质量检查报告")
    lines.append("")
    lines.append(f"- 生成时间：{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 文件数量：{summary['file_count']}")
    lines.append("")
    lines.append("## 文件统计")
    lines.append("")
    lines.append("| 文件 | 行数 | 列数 | 缺失率 | 日期范围 | 重复键 | 读取状态 |")
    lines.append("| --- | ---: | ---: | ---: | --- | --- | --- |")
    for item in summary["files"]:
        date_str = ", ".join(
            f"{k}: {v['min']}~{v['max']}" for k, v in item.get("dates", {}).items()
        ) or "-"
        dup_str = ", ".join(
            f"{k}: {v}" for k, v in item.get("duplicates", {}).items()
        ) or "-"
        lines.append(
            f"| {item['path']} | {item['rows']} | {item['cols']} | "
            f"{item['missing_ratio']} | {date_str} | {dup_str} | {item['status']} |"
        )
    lines.append("")
    lines.append("## 异常与建议")
    lines.append("")
    lines.append("- 缺失率过高/重复键异常的文件：手动复核数据源与落库逻辑。")
    lines.append("- 日期范围异常的文件：检查抓取/更新频率是否遗漏。")
    lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Data quality checker")
    parser.add_argument("--root", default="data", help="数据根目录")
    parser.add_argument("--out", default="", help="输出报告路径（Markdown）")
    parser.add_argument("--json", default="", help="输出JSON摘要路径")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    files = list_data_files(root)
    results = []
    for path in files:
        df = read_file(path)
        if df is None:
            results.append(
                {
                    "path": str(path),
                    "rows": 0,
                    "cols": 0,
                    "missing_ratio": 0,
                    "dates": {},
                    "duplicates": {},
                    "status": "read_failed",
                }
            )
            continue
        summary = summarize_df(df)
        results.append(
            {
                "path": str(path),
                "rows": summary["rows"],
                "cols": summary["cols"],
                "missing_ratio": summary["missing_ratio"],
                "dates": summary["dates"],
                "duplicates": summary["duplicates"],
                "missing_top10": summary["missing_top10"],
                "status": "ok",
            }
        )

    report = {"file_count": len(results), "files": results}

    if args.out:
        render_report(report, Path(args.out))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Scanned files: {len(results)}")
    failed = sum(1 for r in results if r["status"] != "ok")
    print(f"Read failed: {failed}")


if __name__ == "__main__":
    main()
