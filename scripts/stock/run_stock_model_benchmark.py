#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sage_core.stock_selection.stock_selector import StockSelector
from scripts.data._shared.runtime import get_data_root
from scripts.stock.run_stock_selector_monthly import (
    _build_selector_config,
    _build_training_panel,
    _evaluate_holdout_predictions,
    _find_latest_trade_date,
    _load_hs300_universe,
    _resolve_tushare_root,
    _split_train_valid_by_date,
)


def _safe_float(value: Any, default: float = float("-inf")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    return number


def _rank_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sorted_rows = sorted(
        results,
        key=lambda item: (
            _safe_float(item.get("top_n_excess_return"), default=float("-inf")),
            _safe_float(item.get("rank_ic_mean"), default=float("-inf")),
            _safe_float(item.get("top_n_win_rate"), default=float("-inf")),
        ),
        reverse=True,
    )
    for index, row in enumerate(sorted_rows, start=1):
        row["benchmark_rank"] = int(index)
    return sorted_rows


def _to_metrics_row(model_type: str, metrics: Dict[str, float], status: str, message: str = "") -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "model_type": model_type,
        "status": status,
        "message": message,
    }
    row.update(metrics)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="选股模型基准对比（Rule/LGBM/XGB）")
    parser.add_argument("--config", type=str, default="config/app/strategy_governance.yaml")
    parser.add_argument("--as-of-date", type=str, default=None, help="训练截止日，YYYYMMDD，默认最新交易日")
    parser.add_argument("--train-lookback-days", type=int, default=900, help="训练回看天数")
    parser.add_argument("--valid-days", type=int, default=120, help="留出验证交易日数")
    parser.add_argument("--eval-top-n", type=int, default=10, help="验证集TopN收益评估口径")
    parser.add_argument("--models", type=str, default="rule,lgbm,xgb", help="模型列表，逗号分隔")
    parser.add_argument("--data-dir", type=str, default=None, help="Tushare数据根目录")
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="输出目录，默认 data/backtest/stock_selector/benchmark",
    )
    args = parser.parse_args()

    model_types = [item.strip().lower() for item in args.models.split(",") if item.strip()]
    if not model_types:
        raise ValueError("models 不能为空")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    base_config = _build_selector_config(config_path)

    data_root = _resolve_tushare_root(args.data_dir)
    latest_trade_date = _find_latest_trade_date(data_root / "daily", args.as_of_date)
    start_date = (pd.to_datetime(latest_trade_date) - timedelta(days=int(args.train_lookback_days))).strftime("%Y%m%d")
    universe = _load_hs300_universe(data_root, latest_trade_date)
    panel = _build_training_panel(data_root, start_date, latest_trade_date, universe)
    if panel.empty:
        raise ValueError("训练面板为空")

    train_panel, valid_panel = _split_train_valid_by_date(panel, int(args.valid_days))
    if train_panel.empty or valid_panel.empty:
        raise ValueError("留出验证集为空，请调大 train-lookback-days 或减小 valid-days")

    benchmark_rows: List[Dict[str, Any]] = []
    for model_type in model_types:
        cfg = replace(base_config, model_type=model_type)
        selector = StockSelector(cfg)
        try:
            selector.fit(train_panel)
            predictions = selector.predict(valid_panel)
            metrics = _evaluate_holdout_predictions(
                panel=panel,
                pred=predictions,
                top_n=int(args.eval_top_n),
                label_horizon=int(cfg.label_horizons[0]) if cfg.label_horizons else 20,
            )
            row = _to_metrics_row(model_type=model_type, metrics=metrics, status="ok")
            row["feature_count"] = int(len(selector.feature_cols or []))
        except ModuleNotFoundError as error:
            row = _to_metrics_row(
                model_type=model_type,
                metrics={},
                status="skipped_dependency",
                message=str(error),
            )
        except Exception as error:  # noqa: BLE001
            row = _to_metrics_row(
                model_type=model_type,
                metrics={},
                status="failed",
                message=str(error),
            )
        benchmark_rows.append(row)

    ranked_rows = _rank_results([row.copy() for row in benchmark_rows if row.get("status") == "ok"])
    rank_by_model = {row["model_type"]: int(row["benchmark_rank"]) for row in ranked_rows}
    for row in benchmark_rows:
        row["benchmark_rank"] = rank_by_model.get(row["model_type"])

    output_root = (
        Path(args.output_root) if args.output_root else (get_data_root() / "backtest" / "stock_selector" / "benchmark")
    )
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_df = pd.DataFrame(benchmark_rows)
    result_path = output_root / f"stock_model_benchmark_{run_id}.parquet"
    summary_path = output_root / f"stock_model_benchmark_{run_id}.json"
    latest_json = output_root / "stock_model_benchmark_latest.json"

    result_df.to_parquet(result_path, index=False)
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_id,
        "as_of_date": latest_trade_date,
        "train_lookback_days": int(args.train_lookback_days),
        "valid_days": int(args.valid_days),
        "eval_top_n": int(args.eval_top_n),
        "models": model_types,
        "train_rows": int(len(train_panel)),
        "valid_rows": int(len(valid_panel)),
        "results": benchmark_rows,
        "winner": ranked_rows[0]["model_type"] if ranked_rows else None,
        "artifacts": {
            "result_parquet": str(result_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"模型基准对比完成: {summary_path}")
    if ranked_rows:
        print(f"当前第一名: {ranked_rows[0]['model_type']}")


if __name__ == "__main__":
    main()
