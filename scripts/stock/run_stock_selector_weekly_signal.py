#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sage_core.stock_selection.stock_selector import SelectionConfig, StockSelector
from scripts.data._shared.runtime import get_data_root
from scripts.stock.run_stock_selector_monthly import (
    _build_training_panel,
    _find_latest_trade_date,
    _latest_weekly_trade_date,
    _load_hs300_universe,
    _resolve_tushare_root,
    _to_yyyymmdd,
)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_latest_model_metadata(models_dir: Path) -> Path:
    candidates = sorted(models_dir.glob("stock_selector_*.meta.json"))
    if not candidates:
        raise FileNotFoundError(f"未找到模型元数据: {models_dir}")
    return candidates[-1]


def _load_selector_from_metadata(metadata_path: Path) -> tuple[StockSelector, Dict[str, Any]]:
    metadata = _load_json(metadata_path)
    cfg_dict = metadata.get("selector_config", {})
    cfg = SelectionConfig(
        model_type=cfg_dict.get("model_type", "rule"),
        label_horizons=tuple(cfg_dict.get("label_horizons", [120])),
        label_weights=tuple(cfg_dict.get("label_weights", [1.0])),
        risk_adjusted=bool(cfg_dict.get("risk_adjusted", True)),
        vol_window=int(cfg_dict.get("vol_window", 20)),
        industry_rank=bool(cfg_dict.get("industry_rank", True)),
        industry_col=cfg_dict.get("industry_col", "industry_l1"),
        rank_mode=cfg_dict.get("rank_mode", "industry"),
        date_col=cfg_dict.get("date_col", "trade_date"),
        code_col=cfg_dict.get("code_col", "ts_code"),
        price_col=cfg_dict.get("price_col", "close"),
        feature_cols=tuple(cfg_dict.get("feature_cols")) if cfg_dict.get("feature_cols") else None,
        rule_weights=cfg_dict.get("rule_weights"),
        min_feature_count=int(cfg_dict.get("min_feature_count", 8)),
        max_feature_count=int(cfg_dict.get("max_feature_count", 30)),
        lgbm_params=cfg_dict.get("lgbm_params", SelectionConfig().lgbm_params),
        xgb_params=cfg_dict.get("xgb_params", SelectionConfig().xgb_params),
    )
    selector = StockSelector(cfg)
    selector.feature_cols = list(metadata.get("feature_cols", []))
    selector.feature_medians = dict(metadata.get("feature_medians", {}))
    model_type = metadata.get("model_type", cfg.model_type)
    model_path = Path(metadata.get("model_path", ""))
    if not model_path.is_absolute():
        model_path = (metadata_path.parent / model_path).resolve()

    if model_type == "rule":
        selector.rule_weights = dict(metadata.get("rule_weights", {}))
    elif model_type == "lgbm":
        import lightgbm as lgb  # type: ignore

        selector.model = lgb.Booster(model_file=str(model_path))
    elif model_type == "xgb":
        import xgboost as xgb  # type: ignore

        model = xgb.XGBRegressor()
        model.load_model(str(model_path))
        selector.model = model
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    selector.is_trained = True
    return selector, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="加载最新模型并导出最新周选股信号")
    parser.add_argument("--as-of-date", type=str, default=None, help="截止日期 YYYYMMDD，默认最新交易日")
    parser.add_argument("--feature-lookback-days", type=int, default=260, help="周信号特征回看天数")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--data-dir", type=str, default=None, help="Tushare数据目录")
    parser.add_argument("--output-root", type=str, default=None, help="默认 data/signals/stock_selector/monthly")
    parser.add_argument("--model-metadata", type=str, default=None, help="显式指定模型元数据 json")
    args = parser.parse_args()

    data_root = _resolve_tushare_root(args.data_dir)
    output_root = (
        Path(args.output_root) if args.output_root else (get_data_root() / "signals" / "stock_selector" / "monthly")
    )
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    if args.model_metadata:
        metadata_path = Path(args.model_metadata)
        if not metadata_path.is_absolute():
            metadata_path = ROOT / metadata_path
    else:
        metadata_path = _find_latest_model_metadata(output_root / "models")

    selector, metadata = _load_selector_from_metadata(metadata_path)

    latest_trade_date = _find_latest_trade_date(data_root / "daily", args.as_of_date)
    start_date = (pd.to_datetime(latest_trade_date) - timedelta(days=int(args.feature_lookback_days))).strftime(
        "%Y%m%d"
    )
    universe = _load_hs300_universe(data_root, latest_trade_date)
    panel = _build_training_panel(data_root, start_date, latest_trade_date, universe)
    if panel.empty:
        raise ValueError("周信号输入面板为空")

    signal_trade_date = _latest_weekly_trade_date(panel["trade_date"].astype(str).tolist())
    signals = selector.select_top(panel, top_n=int(args.top_n), trade_date=signal_trade_date).copy()
    signals["trade_date"] = _to_yyyymmdd(signal_trade_date)
    signals["model_version"] = metadata.get("model_version", "unknown")
    if "confidence" not in signals.columns:
        signals["confidence"] = signals["rank"].rank(pct=True)

    signal_path = output_root / f"weekly_signals_{signal_trade_date}.parquet"
    summary_path = output_root / f"weekly_signal_summary_{signal_trade_date}.json"
    signals.to_parquet(signal_path, index=False)
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "signal_trade_date": signal_trade_date,
        "as_of_date": latest_trade_date,
        "top_n": int(args.top_n),
        "rows": int(len(signals)),
        "model_metadata": str(metadata_path),
        "signal_file": str(signal_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"周信号更新完成: {signal_path}")
    print(f"摘要: {summary_path}")


if __name__ == "__main__":
    main()
