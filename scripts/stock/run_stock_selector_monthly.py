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

from sage_core.stock_selection.stock_selector import SelectionConfig, StockSelector
from scripts.data._shared.runtime import get_data_root, get_tushare_root


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


def _parquet_columns(path: Path) -> List[str]:
    if not path.exists():
        return []
    try:
        import pyarrow.parquet as pq  # type: ignore

        return list(pq.ParquetFile(path).schema.names)
    except Exception:
        try:
            return list(pd.read_parquet(path).columns)
        except Exception:
            return []


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


def _load_all_stocks_universe(data_root: Path, trade_date: str, exclude_st: bool = True) -> List[str]:
    """
    加载全市场股票池（支持 ST 过滤）

    Args:
        data_root: 数据根目录
        trade_date: 交易日期
        exclude_st: 是否剔除 ST 股票

    Returns:
        股票代码列表
    """
    # 优先使用已过滤的股票列表
    filtered_path = data_root / "metadata" / "filtered_stocks_list.csv"
    all_path = data_root / "metadata" / "all_stocks_list.csv"

    if exclude_st and filtered_path.exists():
        df = pd.read_csv(filtered_path)
        if "tushare_code" in df.columns:
            return sorted(df["tushare_code"].astype(str).unique().tolist())

    if all_path.exists():
        df = pd.read_csv(all_path)
        if "tushare_code" in df.columns:
            return sorted(df["tushare_code"].astype(str).unique().tolist())

    # 回退到从 daily_basic 提取
    path = data_root / "daily_basic_all.parquet"
    if not path.exists():
        raise FileNotFoundError(f"未找到 daily_basic 文件: {path}")

    df = pd.read_parquet(path, columns=["ts_code", "trade_date"])
    df["trade_date"] = df["trade_date"].astype(str)

    # 只取交易日期附近的股票
    valid = df[df["trade_date"] <= trade_date]
    if valid.empty:
        raise ValueError(f"在 {trade_date} 前无可用股票数据")

    # 取最近有数据的股票
    recent_dates = valid["trade_date"].sort_values().unique()
    if len(recent_dates) < 5:
        recent_stocks = valid["ts_code"].unique()
    else:
        # 取最近5个交易日都有数据的股票
        recent_5_dates = recent_dates[-5:]
        recent_stocks = (
            valid[valid["trade_date"].isin(recent_5_dates)]
            .groupby("ts_code")
            .filter(lambda x: x["trade_date"].nunique() >= 3)["ts_code"]
            .unique()
        )

    return sorted(recent_stocks.tolist())


def _load_st_stock_list(data_root: Path, trade_date: str) -> set:
    """
    加载指定日期的 ST 股票列表

    通过 Tushare 股票基本信息判断 ST 状态

    Args:
        data_root: 数据根目录
        trade_date: 交易日期

    Returns:
        ST 股票代码集合
    """
    # 检查是否有本地的 ST 股票列表
    st_path = data_root / "metadata" / "st_stocks.csv"
    if st_path.exists():
        df = pd.read_csv(st_path)
        if "tushare_code" in df.columns:
            return set(df["tushare_code"].astype(str).tolist())

    # 从 filtered_stocks_list 和 all_stocks_list 的差异推断 ST
    filtered_path = data_root / "metadata" / "filtered_stocks_list.csv"
    all_path = data_root / "metadata" / "all_stocks_list.csv"

    if filtered_path.exists() and all_path.exists():
        filtered_df = pd.read_csv(filtered_path)
        all_df = pd.read_csv(all_path)

        filtered_codes = (
            set(filtered_df["tushare_code"].astype(str).tolist()) if "tushare_code" in filtered_df.columns else set()
        )
        all_codes = set(all_df["tushare_code"].astype(str).tolist()) if "tushare_code" in all_df.columns else set()

        # 差集即为被过滤的股票（可能包含 ST）
        excluded_codes = all_codes - filtered_codes
        return excluded_codes

    return set()


def _filter_st_and_new_stocks(
    universe: List[str], st_stocks: set, daily_panel: pd.DataFrame, min_list_days: int = 60
) -> List[str]:
    """
    过滤 ST 股票和上市不足 N 天的新股

    Args:
        universe: 原始股票池
        st_stocks: ST 股票集合
        daily_panel: 日线数据
        min_list_days: 最小上市天数

    Returns:
        过滤后的股票池
    """
    # 1. 剔除 ST
    filtered = [code for code in universe if code not in st_stocks]

    # 2. 剔除上市不足 min_list_days 的新股
    if not daily_panel.empty and len(filtered) > 0:
        daily_filtered = daily_panel[daily_panel["ts_code"].isin(filtered)]
        stock_trade_counts = daily_filtered.groupby("ts_code").size()
        valid_stocks = stock_trade_counts[stock_trade_counts >= min_list_days].index.tolist()
        filtered = [code for code in filtered if code in valid_stocks]

    return filtered


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
        "ts_code",
        "trade_date",
        "turnover_rate_f",
        "pe_ttm",
        "pb",
        "total_mv",
        "dv_ttm",
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
        df.sort_values(["ts_code", "trade_date"]).groupby("ts_code")["northbound_net_flow"].diff()
    )
    return df[["ts_code", "trade_date", "northbound_hold_ratio", "northbound_net_flow"]]


def _load_industry_map(data_root: Path, as_of_trade_date: str) -> pd.DataFrame:
    path = data_root / "northbound" / "sw_constituents.parquet"
    available_cols = set(_parquet_columns(path))
    code_col = "con_code" if "con_code" in available_cols else ("ts_code" if "ts_code" in available_cols else None)
    if code_col is None or "industry_name" not in available_cols:
        return pd.DataFrame(columns=["ts_code", "industry_l1"])

    cols = [code_col, "industry_name"]
    if "in_date" in available_cols:
        cols.append("in_date")
    if "out_date" in available_cols:
        cols.append("out_date")
    df = _read_parquet(path, columns=cols)
    if df.empty:
        return pd.DataFrame(columns=["ts_code", "industry_l1"])

    if "in_date" in df.columns:
        as_of = _to_yyyymmdd(as_of_trade_date)
        df["in_date"] = df["in_date"].astype(str)
        if "out_date" in df.columns:
            df["out_date"] = df["out_date"].fillna("").astype(str)
        else:
            df["out_date"] = ""
        active = df[(df["in_date"] <= as_of) & ((df["out_date"] == "") | (df["out_date"] > as_of))].copy()
    else:
        active = df.copy()
    if active.empty:
        return pd.DataFrame(columns=["ts_code", "industry_l1"])

    sort_cols = [code_col]
    if "in_date" in active.columns:
        sort_cols.append("in_date")
    active = active.sort_values(sort_cols).groupby(code_col, as_index=False).tail(1)
    active = active.rename(columns={code_col: "ts_code", "industry_name": "industry_l1"})
    active["ts_code"] = active["ts_code"].astype(str)
    active["industry_l1"] = active["industry_l1"].astype(str)
    active = active.dropna(subset=["ts_code", "industry_l1"]).drop_duplicates(subset=["ts_code"], keep="last")
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
        "ts_code",
        "ann_date",
        "roe_dt",
        "grossprofit_margin",
        "netprofit_yoy",
        "dt_netprofit_yoy",
        "debt_to_assets",
        "ocfps",
        "roic",
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
    out_cols = [
        "ts_code",
        "trade_date",
        "roe_dt",
        "grossprofit_margin",
        "netprofit_yoy",
        "debt_to_assets",
        "ocfps",
        "roic",
    ]
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


def _fit_selector_with_fallback(
    selector_cfg: SelectionConfig,
    train_df: pd.DataFrame,
    allow_rule_fallback: bool = False,
) -> tuple[StockSelector, SelectionConfig, bool]:
    selector = StockSelector(selector_cfg)
    effective_cfg = selector_cfg
    fallback_used = False
    try:
        selector.fit(train_df)
    except ModuleNotFoundError:
        if not allow_rule_fallback:
            raise
        effective_cfg = replace(selector_cfg, model_type="rule")
        selector = StockSelector(effective_cfg)
        selector.fit(train_df)
        fallback_used = True
    return selector, effective_cfg, fallback_used


def _calc_future_return(df: pd.DataFrame, horizon: int = 20) -> pd.Series:
    out = df[["ts_code", "trade_date", "close"]].copy()
    out = out.sort_values(["ts_code", "trade_date"])
    future_ret = out.groupby("ts_code")["close"].shift(-horizon) / out["close"] - 1.0
    return future_ret.astype(float)


def _split_train_valid_by_date(df: pd.DataFrame, valid_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if valid_days <= 0:
        return df.copy(), pd.DataFrame(columns=df.columns)
    dates = sorted(df["trade_date"].astype(str).unique().tolist())
    if len(dates) <= valid_days:
        return pd.DataFrame(columns=df.columns), df.copy()
    split_point = dates[-valid_days]
    train_df = df[df["trade_date"].astype(str) < split_point].copy()
    valid_df = df[df["trade_date"].astype(str) >= split_point].copy()
    return train_df, valid_df


def _evaluate_holdout_predictions(
    panel: pd.DataFrame,
    pred: pd.DataFrame,
    top_n: int = 10,
    label_horizon: int = 20,
) -> Dict[str, float]:
    if panel.empty or pred.empty:
        return {}

    eval_df = pred.copy()
    ret20 = _calc_future_return(panel, horizon=label_horizon)
    panel_key = panel[["ts_code", "trade_date"]].copy()
    panel_key["future_return"] = ret20
    eval_df = eval_df.merge(panel_key, on=["ts_code", "trade_date"], how="left")
    eval_df["future_return"] = pd.to_numeric(eval_df["future_return"], errors="coerce")
    eval_df["score"] = pd.to_numeric(eval_df["score"], errors="coerce")
    eval_df = eval_df.dropna(subset=["score", "future_return"])
    if eval_df.empty:
        return {}

    daily_rank_ic = (
        eval_df.groupby("trade_date")
        .apply(
            lambda g: g["score"]
            .rank(pct=True)
            .corr(
                g["future_return"].rank(pct=True),
                method="pearson",
            )
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    top_daily = (
        eval_df.sort_values(["trade_date", "score"], ascending=[True, False])
        .groupby("trade_date", as_index=False)
        .head(max(int(top_n), 1))
        .groupby("trade_date", as_index=False)["future_return"]
        .mean()
        .rename(columns={"future_return": "top_return"})
    )
    all_daily = (
        eval_df.groupby("trade_date", as_index=False)["future_return"]
        .mean()
        .rename(columns={"future_return": "all_return"})
    )
    compare = top_daily.merge(all_daily, on="trade_date", how="inner")
    compare["excess_return"] = compare["top_return"] - compare["all_return"]
    if compare.empty:
        return {}

    ic_mean = float(daily_rank_ic.mean()) if not daily_rank_ic.empty else float("nan")
    ic_std = float(daily_rank_ic.std()) if len(daily_rank_ic) > 1 else float("nan")
    return {
        "days": float(compare["trade_date"].nunique()),
        "rank_ic_mean": ic_mean,
        "rank_ic_std": ic_std,
        "rank_ic_ir": (
            float(ic_mean / (ic_std + 1e-12)) if np.isfinite(ic_mean) and np.isfinite(ic_std) else float("nan")
        ),
        "top_n_avg_return": float(compare["top_return"].mean()),
        "universe_avg_return": float(compare["all_return"].mean()),
        "top_n_excess_return": float(compare["excess_return"].mean()),
        "top_n_win_rate": float((compare["excess_return"] > 0).mean()),
    }


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
        rank_mode=raw.get("rank_mode", "industry"),
        industry_rank=bool(raw.get("industry_rank", True)),
        lgbm_params=raw.get("lgbm_params", SelectionConfig().lgbm_params),
        xgb_params=raw.get("xgb_params", SelectionConfig().xgb_params),
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
    parser.add_argument("--valid-days", type=int, default=120, help="训练后留出验证交易日数量（0表示关闭）")
    parser.add_argument("--eval-top-n", type=int, default=10, help="验证集TopN收益评估口径")
    parser.add_argument("--data-dir", type=str, default=None, help="Tushare数据根目录")
    parser.add_argument(
        "--output-root", type=str, default=None, help="输出目录，默认 data/signals/stock_selector/monthly"
    )
    parser.add_argument(
        "--universe",
        type=str,
        default="hs300",
        choices=["hs300", "all"],
        help="股票池：hs300=沪深300, all=全市场（过滤ST）",
    )
    parser.add_argument("--exclude-st", action="store_true", default=True, help="剔除ST股票（仅对 all 生效）")
    parser.add_argument("--min-list-days", type=int, default=60, help="最小上市天数（新股过滤）")
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

    # 根据参数选择股票池
    if args.universe == "hs300":
        universe = _load_hs300_universe(data_root, end_date)
        print(f"股票池：沪深300，共 {len(universe)} 只股票")
    else:
        universe = _load_all_stocks_universe(data_root, end_date, exclude_st=args.exclude_st)
        print(f"股票池：全市场，共 {len(universe)} 只股票（ST过滤={'开启' if args.exclude_st else '关闭'}）")

        # 加载 ST 股票并进一步过滤
        if args.exclude_st:
            st_stocks = _load_st_stock_list(data_root, end_date)
            if st_stocks:
                # 先加载日线数据用于新股过滤
                daily_panel = _load_daily_panel(data_root, start_date, end_date, universe)
                universe = _filter_st_and_new_stocks(universe, st_stocks, daily_panel, min_list_days=args.min_list_days)
                print(f"ST/新股过滤后：{len(universe)} 只股票")

    panel = _build_training_panel(data_root, start_date, end_date, universe)
    if panel.empty:
        raise ValueError("训练面板为空")

    train_panel, valid_panel = _split_train_valid_by_date(panel, int(args.valid_days))
    validation_metrics: Dict[str, float] = {}
    validation_prediction_path = None
    if not train_panel.empty and not valid_panel.empty:
        eval_selector, _, _ = _fit_selector_with_fallback(
            selector_cfg=selector_cfg,
            train_df=train_panel,
            allow_rule_fallback=args.allow_rule_fallback,
        )
        valid_pred = eval_selector.predict(valid_panel)
        validation_metrics = _evaluate_holdout_predictions(
            panel=panel,
            pred=valid_pred,
            top_n=int(args.eval_top_n),
            label_horizon=int(selector_cfg.label_horizons[0]) if selector_cfg.label_horizons else 20,
        )
    else:
        valid_pred = pd.DataFrame()

    selector, effective_cfg, fallback_used = _fit_selector_with_fallback(
        selector_cfg=selector_cfg,
        train_df=panel,
        allow_rule_fallback=args.allow_rule_fallback,
    )

    signal_trade_date = _latest_weekly_trade_date(panel["trade_date"].unique().tolist())
    weekly_signals = selector.select_top(panel, top_n=int(args.top_n), trade_date=signal_trade_date).copy()
    weekly_signals["trade_date"] = signal_trade_date

    model_version = f"stock_selector_{effective_cfg.model_type}_{end_date}"
    weekly_signals["model_version"] = model_version
    if "confidence" not in weekly_signals.columns:
        weekly_signals["confidence"] = weekly_signals["rank"].rank(pct=True)

    importance = _extract_feature_importance(selector)

    output_root = (
        Path(args.output_root) if args.output_root else (get_data_root() / "signals" / "stock_selector" / "monthly")
    )
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    model_dir = output_root / "models"

    signal_path = output_root / f"weekly_signals_{signal_trade_date}.parquet"
    importance_path = output_root / f"feature_importance_{end_date}.parquet"
    summary_path = output_root / f"training_summary_{end_date}.json"
    validation_pred_path = output_root / f"validation_predictions_{end_date}.parquet"
    model_path = model_dir / f"{model_version}"
    metadata_path = model_dir / f"{model_version}.meta.json"

    weekly_signals.to_parquet(signal_path, index=False)
    importance.to_parquet(importance_path, index=False)
    if not valid_pred.empty:
        valid_pred.to_parquet(validation_pred_path, index=False)
        validation_prediction_path = str(validation_pred_path)
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
        "validation": {
            "valid_days": int(args.valid_days),
            "eval_top_n": int(args.eval_top_n),
            "metrics": validation_metrics,
            "prediction_file": validation_prediction_path,
        },
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
