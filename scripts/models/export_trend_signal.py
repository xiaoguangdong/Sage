#!/usr/bin/env python3
"""
导出趋势主信号（默认：日级别均线确认）

用法:
  python scripts/models/export_trend_signal.py
  python scripts/models/export_trend_signal.py --timeframe daily --output-dir data/signals
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.models.label_hs300_daily_weekly import HS300Labeler


def export_trend_main_signal(timeframe: str, output_dir: Path, data_dir: str | None = None) -> Path:
    if timeframe != "daily":
        raise ValueError("当前仅支持日级别导出，周线请后续启用")

    labeler = HS300Labeler(timeframe=timeframe, data_dir=data_dir)
    labeler.calculate_indicators()
    labeler.label_ma_confirmation()
    labeler.label_main()

    df = labeler.df.copy()
    df["trade_date"] = df["date"].dt.strftime("%Y-%m-%d")
    df["state"] = df["label_main"].map({2: "RISK_ON", 1: "NEUTRAL", 0: "RISK_OFF"})

    output_dir.mkdir(parents=True, exist_ok=True)
    as_of = df["date"].iloc[-1].strftime("%Y%m%d")
    output_file = output_dir / f"trend_state_{as_of}.parquet"

    output_cols = [
        "trade_date",
        "state",
        "label_main",
        "label_main_confidence",
        "close",
    ]
    df[output_cols].to_parquet(output_file, index=False)

    return output_file


def summarize(output_file: Path) -> None:
    df = pd.read_parquet(output_file)
    counts = df["state"].value_counts()
    total = len(df)
    conf = df["label_main_confidence"]

    print("导出完成")
    print(f"文件: {output_file}")
    print(f"样本数: {total}")
    for state in ["RISK_ON", "NEUTRAL", "RISK_OFF"]:
        n = int(counts.get(state, 0))
        pct = n / total * 100 if total else 0
        print(f"{state}: {n} ({pct:.1f}%)")
    print(
        f"置信度: mean={conf.mean():.3f}, "
        f"p50={conf.quantile(0.5):.3f}, "
        f"p90={conf.quantile(0.9):.3f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", default="daily", choices=["daily"])
    parser.add_argument("--output-dir", default="data/signals")
    parser.add_argument("--data-dir", default=None, help="数据根目录（tushare）")
    args = parser.parse_args()

    output_file = export_trend_main_signal(args.timeframe, Path(args.output_dir), args.data_dir)
    summarize(output_file)


if __name__ == "__main__":
    main()
