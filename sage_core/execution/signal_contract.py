from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


BASE_SIGNAL_COLUMNS = [
    "trade_date",
    "ts_code",
    "score",
    "rank",
    "confidence",
    "model_version",
]

CONTRACT_COLUMNS = [
    "trade_date",
    "strategy_id",
    "is_champion",
    "signal_name",
    "source",
    "ts_code",
    "score",
    "rank",
    "confidence",
    "model_version",
]


def _ensure_base_schema(frame: pd.DataFrame, strategy_id: str, trade_date: str, is_champion: bool) -> pd.DataFrame:
    missing = [col for col in BASE_SIGNAL_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"信号缺少必要字段: {missing}")
    out = frame[BASE_SIGNAL_COLUMNS].copy()
    out["trade_date"] = str(trade_date)
    out["strategy_id"] = str(strategy_id)
    out["is_champion"] = bool(is_champion)
    out["signal_name"] = "stock_score"
    out["source"] = "champion_challenger"
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out["rank"] = pd.to_numeric(out["rank"], errors="coerce")
    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce").clip(lower=0.0, upper=1.0)
    out = out.dropna(subset=["ts_code", "score", "rank", "confidence"])
    return out[CONTRACT_COLUMNS].reset_index(drop=True)


def build_stock_signal_contract(
    trade_date: str,
    champion_id: str,
    champion_signals: pd.DataFrame,
    challenger_signals: Optional[Dict[str, pd.DataFrame]] = None,
    include_challengers: bool = True,
) -> pd.DataFrame:
    frames = [_ensure_base_schema(champion_signals, champion_id, trade_date, is_champion=True)]
    if include_challengers and challenger_signals:
        for strategy_id, frame in challenger_signals.items():
            if frame is None or frame.empty:
                continue
            frames.append(_ensure_base_schema(frame, strategy_id, trade_date, is_champion=False))
    if not frames:
        return pd.DataFrame(columns=CONTRACT_COLUMNS)
    contract = pd.concat(frames, ignore_index=True)
    contract = contract.sort_values(["is_champion", "strategy_id", "rank"], ascending=[False, True, True])
    return contract.reset_index(drop=True)


def select_champion_signals(contract_df: pd.DataFrame, champion_id: str, min_confidence: float = 0.0) -> pd.DataFrame:
    if contract_df is None or contract_df.empty:
        return pd.DataFrame(columns=CONTRACT_COLUMNS)
    required = set(CONTRACT_COLUMNS)
    if not required.issubset(contract_df.columns):
        raise ValueError(f"contract缺少字段: {sorted(required - set(contract_df.columns))}")

    selected = contract_df[
        (contract_df["strategy_id"] == champion_id)
        & (contract_df["is_champion"] == True)
        & (contract_df["confidence"] >= float(min_confidence))
    ].copy()
    return selected.sort_values("rank").reset_index(drop=True)


def build_stock_industry_map_from_features(df_features: pd.DataFrame) -> pd.DataFrame:
    if df_features is None or df_features.empty:
        return pd.DataFrame(columns=["ts_code", "industry_l1"])

    source = df_features.copy()
    if "ts_code" not in source.columns and "code" in source.columns:
        source["ts_code"] = source["code"]

    industry_col = None
    for col in ("industry_l1", "industry", "sector"):
        if col in source.columns:
            industry_col = col
            break
    if industry_col is None or "ts_code" not in source.columns:
        return pd.DataFrame(columns=["ts_code", "industry_l1"])

    if "trade_date" in source.columns:
        source = source.sort_values("trade_date")
    source = source.dropna(subset=["ts_code", industry_col])
    source = source.groupby("ts_code", as_index=False).tail(1)
    source = source.rename(columns={industry_col: "industry_l1"})
    source["ts_code"] = source["ts_code"].astype(str)
    source["industry_l1"] = source["industry_l1"].astype(str)
    return source[["ts_code", "industry_l1"]].drop_duplicates()


def apply_industry_overlay(
    stock_signals: pd.DataFrame,
    industry_snapshot: pd.DataFrame,
    stock_industry_map: pd.DataFrame,
    signal_weights: Optional[Dict[str, float]] = None,
    overlay_strength: float = 0.20,
) -> pd.DataFrame:
    if stock_signals is None or stock_signals.empty:
        return pd.DataFrame(columns=list(stock_signals.columns) + ["industry_l1", "industry_overlay", "score_raw", "score_final"])

    out = stock_signals.copy()
    out["score_raw"] = pd.to_numeric(out["score"], errors="coerce").fillna(0.0)
    out["score_final"] = out["score_raw"]

    if industry_snapshot is None or industry_snapshot.empty or stock_industry_map is None or stock_industry_map.empty:
        return out

    weights = signal_weights or {
        "policy_score": 0.4,
        "concept_bias": 0.3,
        "northbound_ratio": 0.3,
    }

    snapshot = industry_snapshot.copy()
    required = {"sw_industry", "signal_name", "score", "confidence"}
    if not required.issubset(snapshot.columns):
        return out

    score_series = pd.to_numeric(snapshot["score"], errors="coerce").fillna(0.0)
    if "score_signed" in snapshot.columns:
        signed_score = pd.to_numeric(snapshot["score_signed"], errors="coerce").fillna(0.0)
    elif "direction" in snapshot.columns and score_series.between(0.0, 1.0).all():
        direction = pd.to_numeric(snapshot["direction"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
        signed_score = direction * (2.0 * (score_series - 0.5).abs())
    elif score_series.between(0.0, 1.0).all():
        signed_score = 2.0 * (score_series - 0.5)
    else:
        signed_score = score_series

    snapshot = snapshot[snapshot["signal_name"].isin(weights.keys())].copy()
    if snapshot.empty:
        return out

    snapshot["score"] = signed_score.loc[snapshot.index].clip(-1.0, 1.0)
    snapshot["confidence"] = pd.to_numeric(snapshot["confidence"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    snapshot["signal_weight"] = snapshot["signal_name"].map(weights).astype(float)
    snapshot["weighted"] = snapshot["score"] * snapshot["confidence"] * snapshot["signal_weight"]

    grouped = snapshot.groupby("sw_industry", as_index=False).agg(
        weighted_sum=("weighted", "sum"),
        weight_sum=("signal_weight", "sum"),
    )
    grouped["industry_overlay"] = grouped["weighted_sum"] / grouped["weight_sum"].replace(0.0, np.nan)
    grouped["industry_overlay"] = grouped["industry_overlay"].fillna(0.0).clip(-1.0, 1.0)
    grouped = grouped.rename(columns={"sw_industry": "industry_l1"})[["industry_l1", "industry_overlay"]]

    map_df = stock_industry_map.copy()
    map_df = map_df.dropna(subset=["ts_code", "industry_l1"])
    map_df["ts_code"] = map_df["ts_code"].astype(str)
    map_df["industry_l1"] = map_df["industry_l1"].astype(str)

    out["ts_code"] = out["ts_code"].astype(str)
    out = out.merge(map_df, on="ts_code", how="left")
    out = out.merge(grouped, on="industry_l1", how="left")
    out["industry_overlay"] = out["industry_overlay"].fillna(0.0)
    out["score_final"] = out["score_raw"] * (1.0 + float(overlay_strength) * out["industry_overlay"])
    return out
