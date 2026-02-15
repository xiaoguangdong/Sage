#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data._shared.runtime import get_data_path


def load_inputs(signals_path: Path, mapping_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not signals_path.exists():
        raise FileNotFoundError(f"未找到概念信号: {signals_path}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"未找到概念行业映射: {mapping_path}")

    signals = pd.read_parquet(signals_path)
    mapping = pd.read_parquet(mapping_path)
    if signals.empty:
        raise ValueError("概念信号为空")
    if mapping.empty:
        raise ValueError("概念行业映射为空")
    return signals, mapping


def build_industry_bias(signals: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    signals = signals.copy()
    mapping = mapping.copy()
    signals["trade_date"] = pd.to_datetime(signals["trade_date"]).dt.floor("D")
    merged = signals.merge(mapping, on="concept_code", how="inner")
    if merged.empty:
        return pd.DataFrame()

    merged["rank_pct"] = merged.groupby("trade_date")["concept_heat_score"].rank(pct=True)
    aggregated = (
        merged.groupby(["trade_date", "sw_industry"])
        .agg(
            concept_count=("concept_code", "nunique"),
            mean_rank_pct=("rank_pct", "mean"),
            mean_heat_score=("concept_heat_score", "mean"),
            overheat_rate=("overheat_flag", "mean"),
        )
        .reset_index()
    )
    aggregated["concept_bias_strength"] = ((aggregated["mean_rank_pct"] - 0.5) * 2.0).clip(-1.0, 1.0)
    count_component = (aggregated["concept_count"].clip(upper=20) / 20.0) * 0.6
    heat_component = ((aggregated["mean_heat_score"] + 1.0) / 2.0).clip(0.0, 1.0) * 0.4
    aggregated["concept_signal_confidence"] = (count_component + heat_component).clip(0.0, 1.0)
    return aggregated.sort_values(["trade_date", "concept_bias_strength"], ascending=[True, False])


def main() -> None:
    parser = argparse.ArgumentParser(description="将概念热度信号聚合为行业偏置信号")
    parser.add_argument("--signals-path", type=str, default=None, help="概念信号路径（默认 data/signals/concept_signals_top10.parquet）")
    parser.add_argument("--mapping-path", type=str, default=None, help="概念行业映射路径（默认 data/processed/concepts/concept_industry_primary.parquet）")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录（默认 data/signals/industry）")
    parser.add_argument("--top-k", type=int, default=10, help="读取的概念TopK文件后缀")
    args = parser.parse_args()

    default_signal_name = f"concept_signals_top{args.top_k}.parquet" if args.top_k else "concept_signals.parquet"
    signals_path = Path(args.signals_path) if args.signals_path else get_data_path("signals", default_signal_name)
    if not signals_path.exists() and args.signals_path is None:
        fallback = get_data_path("signals", "concept_signals.parquet")
        if fallback.exists():
            signals_path = fallback
    mapping_path = Path(args.mapping_path) if args.mapping_path else get_data_path(
        "processed", "concepts", "concept_industry_primary.parquet"
    )
    output_dir = Path(args.output_dir) if args.output_dir else get_data_path("signals", "industry", ensure=True)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    signals, mapping = load_inputs(signals_path, mapping_path)
    industry_bias = build_industry_bias(signals, mapping)
    if industry_bias.empty:
        print("未生成行业概念偏置信号")
        return

    output_path = output_dir / "industry_concept_bias.parquet"
    industry_bias.to_parquet(output_path, index=False)
    print(f"已保存: {output_path}")


if __name__ == "__main__":
    main()
