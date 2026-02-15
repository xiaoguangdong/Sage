#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.data._shared.runtime import get_data_root, get_tushare_root
from sage_core.stock_selection.stock_selector import SelectionConfig, StockSelector


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _to_yyyymmdd(value: str | datetime) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y%m%d")
    return pd.to_datetime(value).strftime("%Y%m%d")


def _year_range(start_date: str, end_date: str) -> Iterable[int]:
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    return range(start_year, end_year + 1)


def _read_parquet(path: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path, columns=list(columns) if columns else None)


def _find_latest_trade_date(daily_dir: Path, as_of_date: Optional[str] = None) -> str:
    candidates = sorted(daily_dir.glob("daily_*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"未找到日线文件: {daily_dir}")

    date_limit = _to_yyyymmdd(as_of_date) if as_of_date else None
    latest: Optional[str] = None
    for path in candidates[::-1]:
        df = _read_parquet(path, columns=["trade_date"])
        if df.empty:
            continue
        series = df["trade_date"].astype(str)
        if date_limit:
            series = series[series <= date_limit]
        if series.empty:
            continue
        max_date = str(series.max())
        if latest is None or max_date > latest:
            latest = max_date
            break
    if latest is None:
        raise ValueError("无法解析最新交易日")
    return latest


def _load_hs300_universe(data_root: Path, trade_date: str) -> List[str]:
    path = data_root / "constituents" / "hs300_constituents_all.parquet"
    df = _read_parquet(path, columns=["trade_date", "con_code"])
    if df.empty:
        raise FileNotFoundError(f"未找到沪深300成分文件: {path}")

    df["trade_date"] = df["trade_date"].astype(str)
    valid = df[df["trade_date"] <= trade_date]
    if valid.empty:
        raise ValueError(f"沪深300成分在 {trade_date} 前无可用数据")
    last_date = valid["trade_date"].max()
    members = sorted(valid[valid["trade_date"] == last_date]["con_code"].astype(str).unique().tolist())
    if not members:
        raise ValueError(f"{last_date} 沪深300成分为空")
    return members


def _load_daily_panel(data_root: Path, start_date: str, end_date: str, universe: Sequence[str]) -> pd.DataFrame:
    daily_dir = data_root / "daily"
    frames: List[pd.DataFrame] = []
    cols = ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"]

    for year in _year_range(start_date, end_date):
        path = daily_dir / f"daily_{year}.parquet"
        if not path.exists():
            continue
        df = _read_parquet(path, columns=cols)
        if df.empty:
            continue
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)]
        df = df[df["ts_code"].isin(universe)]
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


def _load_daily_basic_panel(data_root: Path, start_date: str, end_date: str, universe: Sequence[str]) -> pd.DataFrame:
    path = data_root / "daily_basic_all.parquet"
    cols = [
        "ts_code", "trade_date", "turnover_rate_f", "pe_ttm", "pb", "total_mv", "dv_ttm",
    ]
    df = _read_parquet(path, columns=cols)
    if df.empty:
        return pd.DataFrame(columns=cols)
    df["trade_date"] = df["trade_date"].astype(str)
    df = df[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)]
    df = df[df["ts_code"].isin(universe)]
    return df


def _load_northbound_panel(data_root: Path, start_date: str, end_date: str, universe: Sequence[str]) -> pd.DataFrame:
    path = data_root / "northbound" / "northbound_hk_hold.parquet"
    cols = ["ts_code", "trade_date", "vol", "ratio"]
    df = _read_parquet(path, columns=cols)
    if df.empty:
        return pd.DataFrame(columns=["ts_code", "trade_date", "northbound_hold_ratio", "northbound_net_flow"])

    df["trade_date"] = df["trade_date"].astype(str)
    df = df[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)]
    df = df[df["ts_code"].isin(universe)].copy()
    if df.empty:
        return pd.DataFrame(columns=["ts_code", "trade_date", "northbound_hold_ratio", "northbound_net_flow"])

    df["northbound_hold_ratio"] = pd.to_numeric(df["ratio"], errors="coerce") / 100.0
    df["northbound_net_flow"] = pd.to_numeric(df["vol"], errors="coerce")
    df["northbound_net_flow"] = (
        df.sort_values(["ts_code", "trade_date"])
        .groupby("ts_code")["northbound_net_flow"]
        .diff()
    )
    return df[["ts_code", "trade_date", "northbound_hold_ratio", "northbound_net_flow"]]


def _load_industry_map(data_root: Path, as_of_trade_date: str) -> pd.DataFrame:
    path = data_root / "northbound" / "sw_constituents.parquet"
    cols = ["con_code", "in_date", "out_date", "industry_name"]
    df = _read_parquet(path, columns=cols)
    if df.empty:
        return pd.DataFrame(columns=["ts_code", "industry_l1"])

    as_of = _to_yyyymmdd(as_of_trade_date)
    df["in_date"] = df["in_date"].astype(str)
    df["out_date"] = df["out_date"].fillna("").astype(str)
    active = df[(df["in_date"] <= as_of) & ((df["out_date"] == "") | (df["out_date"] > as_of))].copy()
    if active.empty:
        return pd.DataFrame(columns=["ts_code", "industry_l1"])
    active = active.sort_values(["con_code", "in_date"]).groupby("con_code", as_index=False).tail(1)
    active = active.rename(columns={"con_code": "ts_code", "industry_name": "industry_l1"})
    return active[["ts_code", "industry_l1"]]


def _load_fina_asof_panel(
    data_root: Path,
    start_date: str,
    end_date: str,
    universe: Sequence[str],
    daily_panel: pd.DataFrame,
) -> pd.DataFrame:
    fina_dir = data_root / "fundamental"
    cols = [
        "ts_code", "ann_date", "roe_dt", "grossprofit_margin", "netprofit_yoy",
        "dt_netprofit_yoy", "debt_to_assets", "ocfps", "roic",
    ]
    frames: List[pd.DataFrame] = []
    start_year = int(start_date[:4]) - 1
    end_year = int(end_date[:4])
    for year in range(start_year, end_year + 1):
        path = fina_dir / f"fina_indicator_{year}.parquet"
        if not path.exists():
            continue
        cols_avail = pd.read_parquet(path).columns
        use_cols = [c for c in cols if c in cols_avail]
        if "ts_code" not in use_cols or "ann_date" not in use_cols:
            continue
        frames.append(_read_parquet(path, columns=use_cols))

    if not frames:
        return pd.DataFrame(columns=["ts_code", "trade_date"])

    fina = pd.concat(frames, ignore_index=True)
    fina = fina[fina["ts_code"].isin(universe)].copy()
    fina["ann_date"] = fina["ann_date"].astype(str)
    fina = fina[(fina["ann_date"] <= end_date)]
    if fina.empty:
        return pd.DataFrame(columns=["ts_code", "trade_date"])

    if "netprofit_yoy" not in fina.columns and "dt_netprofit_yoy" in fina.columns:
        fina["netprofit_yoy"] = pd.to_numeric(fina["dt_netprofit_yoy"], errors="coerce")
    for col in ["roe_dt", "grossprofit_margin", "netprofit_yoy", "debt_to_assets", "ocfps", "roic"]:
        if col in fina.columns:
            fina[col] = pd.to_numeric(fina[col], errors="coerce")
    if "roic" not in fina.columns:
        fina["roic"] = np.nan

    left = daily_panel[["ts_code", "trade_date"]].drop_duplicates().copy()
    left["trade_date_dt"] = pd.to_datetime(left["trade_date"])
    right = fina.copy()
    right["ann_date_dt"] = pd.to_datetime(right["ann_date"])

    merged_chunks: List[pd.DataFrame] = []
    right_grouped = {code: part.sort_values("ann_date_dt") for code, part in right.groupby("ts_code")}
    for code, left_part in left.groupby("ts_code"):
        left_sorted = left_part.sort_values("trade_date_dt")
        right_sorted = right_grouped.get(code)
        if right_sorted is None or right_sorted.empty:
            chunk = left_sorted.copy()
        else:
            chunk = pd.merge_asof(
                left_sorted,
                right_sorted,
                left_on="trade_date_dt",
                right_on="ann_date_dt",
                direction="backward",
                allow_exact_matches=True,
            )
            chunk["ts_code"] = code
        merged_chunks.append(chunk)

    merged = pd.concat(merged_chunks, ignore_index=True) if merged_chunks else left.copy()
    out_cols = ["ts_code", "trade_date", "roe_dt", "grossprofit_margin", "netprofit_yoy", "debt_to_assets", "ocfps", "roic"]
    for col in out_cols:
        if col not in merged.columns:
            merged[col] = np.nan
    return merged[out_cols]


def _build_training_panel(
    tushare_root: Path,
    start_date: str,
    end_date: str,
    universe: Sequence[str],
) -> pd.DataFrame:
    daily = _load_daily_panel(tushare_root, start_date, end_date, universe)
    if daily.empty:
        raise ValueError("日线数据为空，无法训练")

    daily_basic = _load_daily_basic_panel(tushare_root, start_date, end_date, universe)
    northbound = _load_northbound_panel(tushare_root, start_date, end_date, universe)
    industry_map = _load_industry_map(tushare_root, end_date)
    fina_asof = _load_fina_asof_panel(tushare_root, start_date, end_date, universe, daily_panel=daily)

    panel = daily.copy()
    panel = panel.merge(daily_basic, on=["ts_code", "trade_date"], how="left")
    panel = panel.merge(northbound, on=["ts_code", "trade_date"], how="left")
    panel = panel.merge(industry_map, on="ts_code", how="left")
    panel = panel.merge(fina_asof, on=["ts_code", "trade_date"], how="left")

    if "turnover_rate_f" in panel.columns:
        panel["turnover"] = pd.to_numeric(panel["turnover_rate_f"], errors="coerce")
    panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return panel


def _latest_weekly_trade_date(trade_dates: Sequence[str]) -> str:
    dates = pd.Series(sorted(set(str(d) for d in trade_dates)))
    dt = pd.to_datetime(dates)
    weekly_last = dt.groupby(dt.dt.strftime("%G%V")).max().sort_values()
    return weekly_last.iloc[-1].strftime("%Y%m%d")


def _extract_feature_importance(selector: StockSelector) -> pd.DataFrame:
    feature_cols = selector.feature_cols or []
    if not feature_cols:
        return pd.DataFrame(columns=["feature", "importance", "importance_norm"])

    if selector.config.model_type == "rule":
        weights = getattr(selector, "rule_weights", {})
        data = [{"feature": col, "importance": float(weights.get(col, 0.0))} for col in feature_cols]
    else:
        model = selector.model
        if hasattr(model, "feature_importance"):
            values = model.feature_importance(importance_type="gain")
            data = [{"feature": col, "importance": float(v)} for col, v in zip(feature_cols, values)]
        elif hasattr(model, "feature_importances_"):
            values = getattr(model, "feature_importances_")
            data = [{"feature": col, "importance": float(v)} for col, v in zip(feature_cols, values)]
        else:
            data = [{"feature": col, "importance": 0.0} for col in feature_cols]

    df = pd.DataFrame(data).sort_values("importance", ascending=False).reset_index(drop=True)
    total = float(df["importance"].sum()) if not df.empty else 0.0
    if total > 0:
        df["importance_norm"] = df["importance"] / total
    else:
        df["importance_norm"] = 0.0
    return df


def _save_model_artifact(selector: StockSelector, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    model = selector.model
    if selector.config.model_type == "lgbm" and hasattr(model, "save_model"):
        path = path.with_suffix(".txt")
        model.save_model(str(path))
        return str(path)
    if selector.config.model_type == "xgb" and hasattr(model, "save_model"):
        path = path.with_suffix(".json")
        model.save_model(str(path))
        return str(path)

    path = path.with_suffix(".json")
    payload = {
        "model_type": selector.config.model_type,
        "feature_cols": selector.feature_cols or [],
        "feature_medians": selector.feature_medians,
        "rule_weights": getattr(selector, "rule_weights", {}),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _build_selector_config(config_path: Path) -> SelectionConfig:
    cfg = _load_yaml(config_path)
    raw = cfg.get("seed_balance_strategy", {})
    return SelectionConfig(
        model_type=raw.get("model_type", "lgbm"),
        label_horizons=tuple(raw.get("label_horizons", [20, 60, 120])),
        label_weights=tuple(raw.get("label_weights", [0.5, 0.3, 0.2])),
        risk_adjusted=bool(raw.get("risk_adjusted", True)),
        industry_col=raw.get("industry_col", "industry_l1"),
    )


def _resolve_tushare_root(data_dir_arg: Optional[str]) -> Path:
    if data_dir_arg:
        path = Path(data_dir_arg)
        if not path.is_absolute():
            path = ROOT / path
        return path

    candidates = [
        get_tushare_root(),
        get_data_root() / "tushare",
        ROOT / "data" / "tushare",
    ]
    for path in candidates:
        if (path / "daily").exists():
            return path
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="月度重训选股模型并导出最新周信号")
    parser.add_argument("--config", type=str, default="config/app/strategy_governance.yaml")
    parser.add_argument("--as-of-date", type=str, default=None, help="训练截止日，YYYYMMDD，默认最新交易日")
    parser.add_argument("--train-lookback-days", type=int, default=900, help="训练回看天数")
    parser.add_argument("--top-n", type=int, default=10, help="最新周信号TopN")
    parser.add_argument("--allow-rule-fallback", action="store_true", help="lgbm/xgb不可用时回退rule")
    parser.add_argument("--data-dir", type=str, default=None, help="Tushare数据根目录")
    parser.add_argument("--output-root", type=str, default=None, help="输出目录，默认 data/signals/stock_selector/monthly")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    selector_cfg = _build_selector_config(config_path)

    data_root = _resolve_tushare_root(args.data_dir)
    latest_trade_date = _find_latest_trade_date(data_root / "daily", args.as_of_date)
    start_dt = pd.to_datetime(latest_trade_date) - timedelta(days=int(args.train_lookback_days))
    start_date = start_dt.strftime("%Y%m%d")
    end_date = latest_trade_date

    universe = _load_hs300_universe(data_root, end_date)
    panel = _build_training_panel(data_root, start_date, end_date, universe)
    if panel.empty:
        raise ValueError("训练面板为空")

    selector = StockSelector(selector_cfg)
    effective_cfg = selector_cfg
    fallback_used = False
    try:
        selector.fit(panel)
    except ModuleNotFoundError:
        if not args.allow_rule_fallback:
            raise
        effective_cfg = replace(selector_cfg, model_type="rule")
        selector = StockSelector(effective_cfg)
        selector.fit(panel)
        fallback_used = True

    signal_trade_date = _latest_weekly_trade_date(panel["trade_date"].unique().tolist())
    weekly_signals = selector.select_top(panel, top_n=int(args.top_n), trade_date=signal_trade_date).copy()
    weekly_signals["trade_date"] = signal_trade_date

    model_version = f"stock_selector_{effective_cfg.model_type}_{end_date}"
    weekly_signals["model_version"] = model_version
    if "confidence" not in weekly_signals.columns:
        weekly_signals["confidence"] = weekly_signals["rank"].rank(pct=True)

    importance = _extract_feature_importance(selector)

    output_root = Path(args.output_root) if args.output_root else (get_data_root() / "signals" / "stock_selector" / "monthly")
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    model_dir = output_root / "models"

    signal_path = output_root / f"weekly_signals_{signal_trade_date}.parquet"
    importance_path = output_root / f"feature_importance_{end_date}.parquet"
    summary_path = output_root / f"training_summary_{end_date}.json"
    model_path = model_dir / f"{model_version}"
    metadata_path = model_dir / f"{model_version}.meta.json"

    weekly_signals.to_parquet(signal_path, index=False)
    importance.to_parquet(importance_path, index=False)
    model_saved_path = _save_model_artifact(selector, model_path)
    metadata = {
        "model_version": model_version,
        "model_type": effective_cfg.model_type,
        "model_path": model_saved_path,
        "selector_config": asdict(effective_cfg),
        "feature_cols": selector.feature_cols or [],
        "feature_medians": selector.feature_medians,
        "rule_weights": getattr(selector, "rule_weights", {}),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_version": model_version,
        "fallback_used": fallback_used,
        "selector_config": asdict(effective_cfg),
        "train_range": {"start_date": start_date, "end_date": end_date},
        "signal_trade_date": signal_trade_date,
        "rows": int(len(panel)),
        "codes": int(panel["ts_code"].nunique()),
        "features": selector.feature_cols or [],
        "artifacts": {
            "model": model_saved_path,
            "model_metadata": str(metadata_path),
            "weekly_signals": str(signal_path),
            "feature_importance": str(importance_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"训练完成: model_version={model_version}")
    print(f"周信号输出: {signal_path}")
    print(f"特征重要性: {importance_path}")
    print(f"训练摘要: {summary_path}")


if __name__ == "__main__":
    main()
