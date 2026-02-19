from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


def _normalize_trade_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y%m%d")


def build_industry_score_series(industry_contract: pd.DataFrame) -> pd.DataFrame:
    required = {"trade_date", "sw_industry", "score", "confidence"}
    missing = required - set(industry_contract.columns)
    if missing:
        raise ValueError(f"industry_contract 缺少字段: {sorted(missing)}")

    source = industry_contract.copy()
    source["trade_date"] = _normalize_trade_date(source["trade_date"])
    source["score"] = pd.to_numeric(source["score"], errors="coerce")
    source["confidence"] = pd.to_numeric(source["confidence"], errors="coerce")
    source = source.dropna(subset=["trade_date", "sw_industry", "score", "confidence"])

    grouped = source.groupby(["trade_date", "sw_industry"], as_index=False).apply(
        lambda frame: pd.Series(
            {
                "score": (
                    float((frame["score"] * frame["confidence"]).sum() / frame["confidence"].sum())
                    if float(frame["confidence"].sum()) > 0
                    else float(frame["score"].mean())
                )
            }
        )
    )
    grouped["rank"] = grouped.groupby("trade_date")["score"].rank(ascending=False, method="first").astype(int)
    return grouped.sort_values(["trade_date", "rank"]).reset_index(drop=True)


def prepare_industry_returns(sw_daily: pd.DataFrame, sw_l1_map: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    required = {"trade_date", "ts_code"}
    missing = required - set(sw_daily.columns)
    if missing:
        raise ValueError(f"sw_daily 缺少字段: {sorted(missing)}")

    source = sw_daily.copy()
    source["trade_date"] = _normalize_trade_date(source["trade_date"])

    if "pct_change" in source.columns:
        source["industry_return"] = pd.to_numeric(source["pct_change"], errors="coerce") / 100.0
    elif "pct_chg" in source.columns:
        source["industry_return"] = pd.to_numeric(source["pct_chg"], errors="coerce") / 100.0
    elif "close" in source.columns:
        source = source.sort_values(["ts_code", "trade_date"])
        source["industry_return"] = source.groupby("ts_code")["close"].pct_change()
    else:
        raise ValueError("sw_daily 需要 pct_change/pct_chg/close 之一")

    if sw_l1_map is not None and not sw_l1_map.empty and {"index_code", "industry_name"}.issubset(sw_l1_map.columns):
        map_df = sw_l1_map[["index_code", "industry_name"]].dropna().drop_duplicates()
        map_df["index_code"] = map_df["index_code"].astype(str)
        source["ts_code"] = source["ts_code"].astype(str)
        source = source.merge(map_df, left_on="ts_code", right_on="index_code", how="left")
        source["sw_industry"] = source["industry_name"]
    elif "name" in source.columns:
        source["sw_industry"] = source["name"]
    else:
        source["sw_industry"] = source["ts_code"].astype(str)

    source = source.dropna(subset=["trade_date", "sw_industry", "industry_return"])
    return source[["trade_date", "sw_industry", "industry_return"]].copy()


def prepare_prosperity_scores(
    sw_daily: pd.DataFrame,
    sw_l1_map: Optional[pd.DataFrame] = None,
    momentum_window: int = 20,
    amount_window: int = 20,
    volatility_window: int = 20,
) -> pd.DataFrame:
    required = {"trade_date", "ts_code"}
    missing = required - set(sw_daily.columns)
    if missing:
        raise ValueError(f"sw_daily 缺少字段: {sorted(missing)}")

    source = sw_daily.copy()
    source["trade_date"] = _normalize_trade_date(source["trade_date"])
    source["ts_code"] = source["ts_code"].astype(str)

    if "pct_change" in source.columns:
        source["industry_return"] = pd.to_numeric(source["pct_change"], errors="coerce") / 100.0
    elif "pct_chg" in source.columns:
        source["industry_return"] = pd.to_numeric(source["pct_chg"], errors="coerce") / 100.0
    elif "close" in source.columns:
        source = source.sort_values(["ts_code", "trade_date"])
        source["industry_return"] = source.groupby("ts_code")["close"].pct_change()
    else:
        raise ValueError("sw_daily 需要 pct_change/pct_chg/close 之一")

    if "amount" in source.columns:
        source["amount"] = pd.to_numeric(source["amount"], errors="coerce")
    else:
        source["amount"] = np.nan

    if sw_l1_map is not None and not sw_l1_map.empty and {"index_code", "industry_name"}.issubset(sw_l1_map.columns):
        map_df = sw_l1_map[["index_code", "industry_name"]].dropna().drop_duplicates()
        map_df["index_code"] = map_df["index_code"].astype(str)
        source = source.merge(map_df, left_on="ts_code", right_on="index_code", how="left")
        source["sw_industry"] = source["industry_name"]
    elif "name" in source.columns:
        source["sw_industry"] = source["name"]
    else:
        source["sw_industry"] = source["ts_code"]

    source = source.dropna(subset=["trade_date", "sw_industry", "industry_return"])
    source = (
        source.groupby(["trade_date", "sw_industry"], as_index=False)
        .agg(industry_return=("industry_return", "mean"), amount=("amount", "mean"))
        .sort_values(["sw_industry", "trade_date"])
    )
    source["trade_date_dt"] = pd.to_datetime(source["trade_date"], format="%Y%m%d", errors="coerce")
    source = source.dropna(subset=["trade_date_dt"]).reset_index(drop=True)

    momentum_w = max(5, int(momentum_window))
    amount_w = max(5, int(amount_window))
    volatility_w = max(5, int(volatility_window))

    source["momentum"] = source.groupby("sw_industry")["industry_return"].transform(
        lambda s: (1.0 + s).rolling(momentum_w, min_periods=max(5, momentum_w // 2)).apply(np.prod, raw=True) - 1.0
    )
    source["volatility"] = source.groupby("sw_industry")["industry_return"].transform(
        lambda s: s.rolling(volatility_w, min_periods=max(5, volatility_w // 2)).std()
    )
    amount_ma = source.groupby("sw_industry")["amount"].transform(
        lambda s: s.rolling(amount_w, min_periods=max(5, amount_w // 2)).mean()
    )
    source["amount_ratio"] = (source["amount"] / amount_ma).replace([np.inf, -np.inf], np.nan)

    def _rank_pct(series: pd.Series, ascending: bool = True) -> pd.Series:
        if series.notna().sum() == 0:
            return pd.Series(np.nan, index=series.index)
        return series.rank(pct=True, ascending=ascending, method="average")

    source["momentum_rank"] = source.groupby("trade_date")["momentum"].transform(_rank_pct)
    source["amount_rank"] = source.groupby("trade_date")["amount_ratio"].transform(_rank_pct)
    source["stability_rank"] = source.groupby("trade_date")["volatility"].transform(
        lambda s: _rank_pct(s, ascending=False)
    )

    source["momentum_rank"] = source["momentum_rank"].fillna(0.5).clip(0.0, 1.0)
    source["amount_rank"] = source["amount_rank"].fillna(0.5).clip(0.0, 1.0)
    source["stability_rank"] = source["stability_rank"].fillna(0.5).clip(0.0, 1.0)

    source["score"] = (
        source["momentum_rank"] * 0.5 + source["amount_rank"] * 0.3 + source["stability_rank"] * 0.2
    ).clip(0.0, 1.0)
    has_momentum = source["momentum"].notna().astype(float)
    has_amount = source["amount_ratio"].notna().astype(float)
    has_vol = source["volatility"].notna().astype(float)
    source["confidence"] = ((has_momentum + has_amount + has_vol) / 3.0).clip(0.0, 1.0)
    source["rank"] = source.groupby("trade_date")["score"].rank(ascending=False, method="first").astype(int)

    return (
        source[["trade_date", "sw_industry", "score", "confidence", "rank"]]
        .sort_values(["trade_date", "rank"])
        .reset_index(drop=True)
    )


def prepare_benchmark_returns(index_df: pd.DataFrame) -> pd.DataFrame:
    source = index_df.copy()
    if "trade_date" not in source.columns:
        if "date" in source.columns:
            source["trade_date"] = source["date"]
        elif "datetime" in source.columns:
            source["trade_date"] = source["datetime"]
        else:
            raise ValueError("index_df 缺少字段: ['trade_date']，且未找到 date/datetime")
    source["trade_date"] = _normalize_trade_date(source["trade_date"])

    if "pct_chg" in source.columns:
        source["benchmark_return"] = pd.to_numeric(source["pct_chg"], errors="coerce") / 100.0
    elif "pct_change" in source.columns:
        source["benchmark_return"] = pd.to_numeric(source["pct_change"], errors="coerce") / 100.0
    elif "close" in source.columns:
        source = source.sort_values("trade_date")
        source["benchmark_return"] = pd.to_numeric(source["close"], errors="coerce").pct_change()
    else:
        raise ValueError("index_df 需要 pct_chg/pct_change/close 之一")

    source = source.dropna(subset=["trade_date", "benchmark_return"])
    return source[["trade_date", "benchmark_return"]].drop_duplicates(subset=["trade_date"])


def prepare_trend_gate(
    index_df: pd.DataFrame,
    neutral_multiplier: float = 0.6,
    risk_off_multiplier: float = 0.2,
) -> pd.DataFrame:
    source = index_df.copy()
    if "trade_date" not in source.columns:
        if "date" in source.columns:
            source["trade_date"] = source["date"]
        elif "datetime" in source.columns:
            source["trade_date"] = source["datetime"]
        else:
            raise ValueError("index_df 缺少字段: ['trade_date']，且未找到 date/datetime")
    source["trade_date"] = _normalize_trade_date(source["trade_date"])

    if "close" in source.columns:
        close = pd.to_numeric(source["close"], errors="coerce")
    else:
        raise ValueError("index_df 需要 close 字段用于趋势门控")

    source = source.sort_values("trade_date").reset_index(drop=True)
    source["close"] = close
    source["ma20"] = source["close"].rolling(20, min_periods=20).mean()
    source["ma60"] = source["close"].rolling(60, min_periods=60).mean()

    gate = pd.Series(1.0, index=source.index)
    bull_mask = (source["close"] > source["ma20"]) & (source["ma20"] > source["ma60"])
    neutral_mask = (source["close"] > source["ma60"]) & (~bull_mask)
    risk_off_mask = ~bull_mask & ~neutral_mask
    gate.loc[neutral_mask] = float(neutral_multiplier)
    gate.loc[risk_off_mask] = float(risk_off_multiplier)

    state = pd.Series("RISK_ON", index=source.index)
    state.loc[neutral_mask] = "NEUTRAL"
    state.loc[risk_off_mask] = "RISK_OFF"
    return pd.DataFrame(
        {
            "trade_date": source["trade_date"],
            "trend_gate": gate.clip(lower=0.0, upper=1.0),
            "trend_state": state,
        }
    ).dropna(subset=["trade_date"])


def prepare_crowding_penalty(
    sw_daily: pd.DataFrame,
    sw_l1_map: Optional[pd.DataFrame] = None,
    z_threshold: float = 1.5,
    penalty_factor: float = 0.8,
    roll_window: int = 20,
) -> pd.DataFrame:
    source = sw_daily.copy()
    if "trade_date" not in source.columns:
        raise ValueError("sw_daily 缺少 trade_date 字段")
    if "ts_code" not in source.columns:
        raise ValueError("sw_daily 缺少 ts_code 字段")
    if "amount" not in source.columns:
        raise ValueError("sw_daily 缺少 amount 字段")

    source["trade_date"] = _normalize_trade_date(source["trade_date"])
    source["ts_code"] = source["ts_code"].astype(str)
    source["amount"] = pd.to_numeric(source["amount"], errors="coerce")
    if "pct_change" in source.columns:
        source["pct_return"] = pd.to_numeric(source["pct_change"], errors="coerce") / 100.0
    elif "pct_chg" in source.columns:
        source["pct_return"] = pd.to_numeric(source["pct_chg"], errors="coerce") / 100.0
    elif "close" in source.columns:
        source = source.sort_values(["ts_code", "trade_date"])
        source["pct_return"] = source.groupby("ts_code")["close"].pct_change()
    else:
        source["pct_return"] = 0.0

    if sw_l1_map is not None and not sw_l1_map.empty and {"index_code", "industry_name"}.issubset(sw_l1_map.columns):
        map_df = sw_l1_map[["index_code", "industry_name"]].dropna().drop_duplicates()
        map_df["index_code"] = map_df["index_code"].astype(str)
        source = source.merge(map_df, left_on="ts_code", right_on="index_code", how="left")
        source["sw_industry"] = source["industry_name"]
    elif "name" in source.columns:
        source["sw_industry"] = source["name"]
    else:
        source["sw_industry"] = source["ts_code"]

    source = source.sort_values(["sw_industry", "trade_date"])
    roll_mean = source.groupby("sw_industry")["amount"].transform(
        lambda s: s.rolling(roll_window, min_periods=5).mean()
    )
    roll_std = source.groupby("sw_industry")["amount"].transform(lambda s: s.rolling(roll_window, min_periods=5).std())
    source["crowding_z"] = (source["amount"] - roll_mean) / roll_std.replace(0.0, np.nan)
    source["crowding_z"] = source["crowding_z"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    source["is_overcrowded"] = (source["crowding_z"] >= float(z_threshold)) & (source["pct_return"] >= 0.03)
    source["crowding_penalty"] = np.where(source["is_overcrowded"], float(penalty_factor), 1.0)

    grouped = source.groupby(["trade_date", "sw_industry"], as_index=False).agg(
        crowding_penalty=("crowding_penalty", "min"),
        crowding_z=("crowding_z", "mean"),
        is_overcrowded=("is_overcrowded", "max"),
    )
    grouped["is_overcrowded"] = grouped["is_overcrowded"].astype(bool)
    grouped["crowding_penalty"] = grouped["crowding_penalty"].clip(lower=0.0, upper=1.0)
    return grouped


def prepare_concept_bias_scores(concept_bias: pd.DataFrame) -> pd.DataFrame:
    required = {"trade_date", "sw_industry"}
    missing = required - set(concept_bias.columns)
    if missing:
        raise ValueError(f"concept_bias 缺少字段: {sorted(missing)}")

    source = concept_bias.copy()
    source["trade_date"] = _normalize_trade_date(source["trade_date"])
    source["sw_industry"] = source["sw_industry"].astype(str)

    if "concept_bias_strength" in source.columns:
        bias = pd.to_numeric(source["concept_bias_strength"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
        source["concept_score"] = ((bias + 1.0) / 2.0).clip(0.0, 1.0)
    elif "concept_score" in source.columns:
        source["concept_score"] = pd.to_numeric(source["concept_score"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    elif "score" in source.columns:
        source["concept_score"] = pd.to_numeric(source["score"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    else:
        source["concept_score"] = 0.5

    if "concept_confidence" in source.columns:
        source["concept_confidence"] = pd.to_numeric(source["concept_confidence"], errors="coerce").fillna(0.0)
    elif "concept_signal_confidence" in source.columns:
        source["concept_confidence"] = pd.to_numeric(source["concept_signal_confidence"], errors="coerce").fillna(0.0)
    elif "confidence" in source.columns:
        source["concept_confidence"] = pd.to_numeric(source["confidence"], errors="coerce").fillna(0.0)
    else:
        source["concept_confidence"] = 0.0
    source["concept_confidence"] = source["concept_confidence"].clip(0.0, 1.0)

    return source[["trade_date", "sw_industry", "concept_score", "concept_confidence"]].dropna(
        subset=["trade_date", "sw_industry"]
    )


def prepare_trend_dominant_state(
    trend_gate: pd.DataFrame,
    lookback_days: int = 5,
    dominance_threshold: float = 0.6,
) -> pd.DataFrame:
    if trend_gate is None or trend_gate.empty:
        return pd.DataFrame(columns=["trade_date", "dominant_state", "dominant_ratio"])
    if "trade_date" not in trend_gate.columns:
        raise ValueError("trend_gate 缺少 trade_date 字段")

    source = trend_gate.copy()
    source["trade_date"] = _normalize_trade_date(source["trade_date"])
    source = (
        source.dropna(subset=["trade_date"])
        .sort_values("trade_date")
        .drop_duplicates(subset=["trade_date"], keep="last")
    )

    if "trend_state" in source.columns:
        state = source["trend_state"].astype(str).str.upper()
    elif "trend_gate" in source.columns:
        gate = pd.to_numeric(source["trend_gate"], errors="coerce").fillna(1.0)
        state = pd.Series("RISK_ON", index=source.index)
        state.loc[gate < 0.4] = "RISK_OFF"
        state.loc[(gate >= 0.4) & (gate < 0.95)] = "NEUTRAL"
    else:
        raise ValueError("trend_gate 缺少 trend_state 或 trend_gate 字段")

    state = state.replace({"BULL": "RISK_ON", "BEAR": "RISK_OFF"})
    state = state.where(state.isin(["RISK_ON", "NEUTRAL", "RISK_OFF"]), "NEUTRAL")
    source["trend_state"] = state

    rows = []
    states = source["trend_state"].tolist()
    dates = source["trade_date"].tolist()
    window = max(1, int(lookback_days))
    threshold = float(dominance_threshold)
    for idx, date in enumerate(dates):
        segment = states[max(0, idx - window + 1) : idx + 1]
        total = len(segment)
        risk_on_ratio = segment.count("RISK_ON") / total
        neutral_ratio = segment.count("NEUTRAL") / total
        risk_off_ratio = segment.count("RISK_OFF") / total
        ratio_map = {"RISK_ON": risk_on_ratio, "NEUTRAL": neutral_ratio, "RISK_OFF": risk_off_ratio}
        dominant_state, dominant_ratio = max(ratio_map.items(), key=lambda item: item[1])
        if dominant_ratio < threshold:
            dominant_state = "NEUTRAL"
        rows.append(
            {
                "trade_date": date,
                "dominant_state": dominant_state,
                "dominant_ratio": float(dominant_ratio),
                "risk_on_ratio": float(risk_on_ratio),
                "neutral_ratio": float(neutral_ratio),
                "risk_off_ratio": float(risk_off_ratio),
            }
        )

    return pd.DataFrame(rows)


def blend_industry_scores_with_concept(
    industry_scores: pd.DataFrame,
    concept_bias: pd.DataFrame,
    dominant_state: pd.DataFrame,
    *,
    risk_on_concept_weight: float = 0.6,
    neutral_concept_weight: float = 0.3,
    risk_off_concept_weight: float = 0.1,
    concept_max_stale_days: int = 7,
) -> pd.DataFrame:
    required_scores = {"trade_date", "sw_industry", "score"}
    missing_scores = required_scores - set(industry_scores.columns)
    if missing_scores:
        raise ValueError(f"industry_scores 缺少字段: {sorted(missing_scores)}")

    scores = industry_scores.copy()
    scores["trade_date"] = _normalize_trade_date(scores["trade_date"])
    scores["trade_date_dt"] = pd.to_datetime(scores["trade_date"], format="%Y%m%d", errors="coerce")
    scores["sw_industry"] = scores["sw_industry"].astype(str)
    scores["score"] = pd.to_numeric(scores["score"], errors="coerce")
    scores = scores.dropna(subset=["trade_date", "trade_date_dt", "sw_industry", "score"]).reset_index(drop=True)
    scores["row_id"] = np.arange(len(scores))

    concept = prepare_concept_bias_scores(concept_bias)
    concept["trade_date_dt"] = pd.to_datetime(concept["trade_date"], format="%Y%m%d", errors="coerce")
    concept = concept.dropna(subset=["trade_date_dt"]).sort_values(["sw_industry", "trade_date_dt"])

    aligned_rows = []
    stale_days = max(1, int(concept_max_stale_days))
    for industry_name, group in scores.groupby("sw_industry"):
        group_sorted = group.sort_values("trade_date_dt")
        concept_group = concept[concept["sw_industry"] == industry_name][
            ["trade_date_dt", "concept_score", "concept_confidence"]
        ]
        if concept_group.empty:
            fill = group_sorted[["row_id"]].copy()
            fill["concept_score"] = 0.5
            fill["concept_confidence"] = 0.0
            aligned_rows.append(fill)
            continue
        aligned = pd.merge_asof(
            group_sorted[["row_id", "trade_date_dt"]],
            concept_group.sort_values("trade_date_dt"),
            on="trade_date_dt",
            direction="backward",
            tolerance=pd.Timedelta(days=stale_days),
        )
        aligned["concept_score"] = pd.to_numeric(aligned["concept_score"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
        aligned["concept_confidence"] = (
            pd.to_numeric(aligned["concept_confidence"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        )
        aligned_rows.append(aligned[["row_id", "concept_score", "concept_confidence"]])

    aligned_all = (
        pd.concat(aligned_rows, ignore_index=True)
        if aligned_rows
        else pd.DataFrame(columns=["row_id", "concept_score", "concept_confidence"])
    )
    scores = scores.merge(aligned_all, on="row_id", how="left")
    scores["concept_score"] = scores["concept_score"].fillna(0.5).clip(0.0, 1.0)
    scores["concept_confidence"] = scores["concept_confidence"].fillna(0.0).clip(0.0, 1.0)

    state_df = dominant_state.copy() if dominant_state is not None else pd.DataFrame()
    if not state_df.empty:
        if "trade_date" not in state_df.columns:
            raise ValueError("dominant_state 缺少 trade_date 字段")
        state_col = "dominant_state" if "dominant_state" in state_df.columns else "trend_state"
        state_df["trade_date"] = _normalize_trade_date(state_df["trade_date"])
        state_df["state_for_blend"] = (
            state_df[state_col].astype(str).str.upper().replace({"BULL": "RISK_ON", "BEAR": "RISK_OFF"})
        )
        state_df["state_for_blend"] = state_df["state_for_blend"].where(
            state_df["state_for_blend"].isin(["RISK_ON", "NEUTRAL", "RISK_OFF"]), "NEUTRAL"
        )
        scores = scores.merge(state_df[["trade_date", "state_for_blend"]], on="trade_date", how="left")
    else:
        scores["state_for_blend"] = "NEUTRAL"
    scores["state_for_blend"] = scores["state_for_blend"].fillna("NEUTRAL")

    weights = {
        "RISK_ON": float(risk_on_concept_weight),
        "NEUTRAL": float(neutral_concept_weight),
        "RISK_OFF": float(risk_off_concept_weight),
    }
    scores["concept_weight"] = (
        scores["state_for_blend"].map(weights).fillna(float(neutral_concept_weight)).clip(0.0, 1.0)
    )
    scores["effective_concept_weight"] = (scores["concept_weight"] * scores["concept_confidence"]).clip(0.0, 1.0)
    scores["base_score"] = scores["score"]
    scores["score"] = (
        scores["base_score"] * (1.0 - scores["effective_concept_weight"])
        + scores["concept_score"] * scores["effective_concept_weight"]
    ).clip(0.0, 1.0)
    scores["rank"] = scores.groupby("trade_date")["score"].rank(ascending=False, method="first").astype(int)
    return scores.drop(columns=["row_id", "trade_date_dt"]).sort_values(["trade_date", "rank"]).reset_index(drop=True)


def _compound_return(series: pd.Series) -> float:
    if series is None or len(series) == 0:
        return np.nan
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(np.prod(1.0 + values.values) - 1.0)


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    if x is None or y is None or len(x) < 3 or len(y) < 3:
        return np.nan
    rank_x = x.rank(pct=True)
    rank_y = y.rank(pct=True)
    corr = rank_x.corr(rank_y, method="pearson")
    return float(corr) if pd.notna(corr) else np.nan


def _max_drawdown(cumulative: Iterable[float]) -> float:
    peak = None
    max_dd = 0.0
    for value in cumulative:
        if peak is None or value > peak:
            peak = value
        if peak and peak != 0:
            drawdown = value / peak - 1.0
            if drawdown < max_dd:
                max_dd = float(drawdown)
    return abs(max_dd)


def evaluate_industry_rotation(
    *,
    industry_scores: pd.DataFrame,
    industry_returns: pd.DataFrame,
    benchmark_returns: Optional[pd.DataFrame] = None,
    top_n: int = 2,
    hold_days: int = 5,
    rebalance_step: int = 5,
    cost_rate: float = 0.005,
    trend_gate: Optional[pd.DataFrame] = None,
    crowding_penalty: Optional[pd.DataFrame] = None,
    enable_exposure_penalty: bool = False,
    exposure_penalty_window: int = 12,
    exposure_penalty_threshold: float = 0.6,
    exposure_penalty_factor: float = 0.85,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    score_required = {"trade_date", "sw_industry", "score"}
    ret_required = {"trade_date", "sw_industry", "industry_return"}
    if not score_required.issubset(industry_scores.columns):
        raise ValueError(f"industry_scores 缺少字段: {sorted(score_required - set(industry_scores.columns))}")
    if not ret_required.issubset(industry_returns.columns):
        raise ValueError(f"industry_returns 缺少字段: {sorted(ret_required - set(industry_returns.columns))}")

    scores = industry_scores.copy()
    returns = industry_returns.copy()
    scores["trade_date"] = _normalize_trade_date(scores["trade_date"])
    returns["trade_date"] = _normalize_trade_date(returns["trade_date"])
    scores = scores.dropna(subset=["trade_date", "sw_industry", "score"])
    returns = returns.dropna(subset=["trade_date", "sw_industry", "industry_return"])

    returns["industry_return"] = pd.to_numeric(returns["industry_return"], errors="coerce")
    returns = returns.dropna(subset=["industry_return"])

    benchmark = None
    if benchmark_returns is not None and not benchmark_returns.empty:
        benchmark = benchmark_returns.copy()
        benchmark["trade_date"] = _normalize_trade_date(benchmark["trade_date"])
        benchmark["benchmark_return"] = pd.to_numeric(benchmark["benchmark_return"], errors="coerce")
        benchmark = benchmark.dropna(subset=["trade_date", "benchmark_return"])

    trend_gate_map: Dict[str, float] = {}
    if trend_gate is not None and not trend_gate.empty:
        gate_df = trend_gate.copy()
        if "trade_date" not in gate_df.columns:
            raise ValueError("trend_gate 缺少 trade_date 字段")
        if "trend_gate" not in gate_df.columns:
            if "gate_multiplier" in gate_df.columns:
                gate_df["trend_gate"] = gate_df["gate_multiplier"]
            else:
                raise ValueError("trend_gate 缺少 trend_gate/gate_multiplier 字段")
        gate_df["trade_date"] = _normalize_trade_date(gate_df["trade_date"])
        gate_df["trend_gate"] = pd.to_numeric(gate_df["trend_gate"], errors="coerce").fillna(1.0).clip(0.0, 1.0)
        trend_gate_map = dict(zip(gate_df["trade_date"], gate_df["trend_gate"]))

    crowding_df = pd.DataFrame()
    if crowding_penalty is not None and not crowding_penalty.empty:
        crowding_df = crowding_penalty.copy()
        if not {"trade_date", "sw_industry", "crowding_penalty"}.issubset(crowding_df.columns):
            raise ValueError("crowding_penalty 缺少字段: trade_date/sw_industry/crowding_penalty")
        crowding_df["trade_date"] = _normalize_trade_date(crowding_df["trade_date"])
        crowding_df["crowding_penalty"] = (
            pd.to_numeric(crowding_df["crowding_penalty"], errors="coerce").fillna(1.0).clip(0.0, 1.0)
        )

    trade_dates = sorted(returns["trade_date"].dropna().unique().tolist())
    date_to_idx = {date: idx for idx, date in enumerate(trade_dates)}

    future_rows = []
    for current_date in trade_dates:
        start_idx = date_to_idx[current_date] + 1
        end_idx = start_idx + int(hold_days)
        if end_idx > len(trade_dates):
            continue
        hold_dates = trade_dates[start_idx:end_idx]
        future_slice = returns[returns["trade_date"].isin(hold_dates)]
        future_group = future_slice.groupby("sw_industry", as_index=False)["industry_return"].apply(_compound_return)
        future_group = future_group.rename(columns={"industry_return": "future_return"})
        future_group["trade_date"] = current_date
        future_rows.append(future_group)

    if not future_rows:
        empty_detail = pd.DataFrame(
            columns=[
                "trade_date",
                "selected_industries",
                "portfolio_return",
                "benchmark_return",
                "excess_return",
                "turnover",
                "cost",
                "net_excess_return",
                "rank_ic",
            ]
        )
        empty_contrib = pd.DataFrame(columns=["sw_industry", "drawdown_contribution", "drawdown_share"])
        empty_summary = {
            "periods": 0,
            "industry_hit_rate": 0.0,
            "industry_excess_total_return": 0.0,
            "industry_excess_annual_return": 0.0,
            "industry_cost_adjusted_total_return": 0.0,
            "industry_cost_adjusted_annual_return": 0.0,
            "industry_turnover_mean": 0.0,
            "industry_max_drawdown": 0.0,
            "industry_rank_ic_mean": 0.0,
            "industry_rank_ic_ir": 0.0,
        }
        return empty_detail, empty_contrib, empty_summary

    future_all = pd.concat(future_rows, ignore_index=True)
    candidate_dates = sorted(set(scores["trade_date"]).intersection(set(future_all["trade_date"])))
    candidate_dates = candidate_dates[:: max(1, int(rebalance_step))]

    detail_rows = []
    drawdown_contrib: Dict[str, float] = {}
    prev_selected: Optional[set[str]] = None
    selected_history: list[set[str]] = []

    for date in candidate_dates:
        day_scores = scores[scores["trade_date"] == date].copy()
        day_future = future_all[future_all["trade_date"] == date].copy()
        merged = day_scores.merge(day_future, on=["trade_date", "sw_industry"], how="inner")
        if merged.empty:
            continue

        merged = merged.sort_values("score", ascending=False)
        selected = merged.head(int(top_n))
        if selected.empty:
            continue

        selected = selected.copy()
        selected["crowding_penalty"] = 1.0
        if not crowding_df.empty:
            selected = selected.merge(
                crowding_df[["trade_date", "sw_industry", "crowding_penalty"]],
                on=["trade_date", "sw_industry"],
                how="left",
                suffixes=("", "_crowd"),
            )
            if "crowding_penalty_crowd" in selected.columns:
                selected["crowding_penalty"] = pd.to_numeric(
                    selected["crowding_penalty_crowd"], errors="coerce"
                ).fillna(1.0)
                selected = selected.drop(columns=["crowding_penalty_crowd"])
            selected["crowding_penalty"] = selected["crowding_penalty"].clip(0.0, 1.0)

        selected["exposure_penalty"] = 1.0
        if enable_exposure_penalty:
            history_window = selected_history[-max(1, int(exposure_penalty_window)) :]
            for idx, row in selected.iterrows():
                industry_name = str(row["sw_industry"])
                if history_window:
                    hit_count = sum(1 for prev_set in history_window if industry_name in prev_set)
                    ratio = hit_count / len(history_window)
                else:
                    ratio = 0.0
                if ratio >= float(exposure_penalty_threshold):
                    selected.at[idx, "exposure_penalty"] = float(exposure_penalty_factor)

        selected["effective_return"] = (
            selected["future_return"] * selected["crowding_penalty"] * selected["exposure_penalty"]
        )
        gate_multiplier = float(trend_gate_map.get(date, 1.0))
        portfolio_return = float(selected["effective_return"].mean()) * gate_multiplier

        hold_start = date_to_idx[date] + 1
        hold_end = hold_start + int(hold_days)
        hold_dates = trade_dates[hold_start:hold_end]
        if benchmark is not None:
            benchmark_slice = benchmark[benchmark["trade_date"].isin(hold_dates)]
            benchmark_return = _compound_return(benchmark_slice["benchmark_return"])
        else:
            benchmark_return = float(day_future["future_return"].mean())
        if pd.isna(benchmark_return):
            benchmark_return = float(day_future["future_return"].mean())

        excess_return = portfolio_return - benchmark_return
        selected_set = set(selected["sw_industry"].astype(str).tolist())
        if prev_selected is None:
            turnover = 1.0
        else:
            overlap = len(prev_selected.intersection(selected_set))
            turnover = 1.0 - overlap / max(1, int(top_n))
        prev_selected = selected_set
        selected_history.append(selected_set)

        cost = float(turnover) * float(cost_rate)
        net_excess_return = excess_return - cost
        rank_ic = _safe_spearman(merged["score"], merged["future_return"])

        for _, row in selected.iterrows():
            contribution = float(row["effective_return"] * gate_multiplier - benchmark_return) / max(1, int(top_n))
            if contribution < 0:
                key = str(row["sw_industry"])
                drawdown_contrib[key] = drawdown_contrib.get(key, 0.0) + abs(contribution)

        detail_rows.append(
            {
                "trade_date": date,
                "selected_industries": "|".join(sorted(selected_set)),
                "portfolio_return": portfolio_return,
                "benchmark_return": benchmark_return,
                "excess_return": excess_return,
                "turnover": turnover,
                "cost": cost,
                "net_excess_return": net_excess_return,
                "rank_ic": rank_ic,
                "trend_gate": gate_multiplier,
                "crowding_penalty_mean": float(selected["crowding_penalty"].mean()),
                "exposure_penalty_mean": float(selected["exposure_penalty"].mean()),
            }
        )

    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        return evaluate_industry_rotation(
            industry_scores=industry_scores.iloc[0:0],
            industry_returns=industry_returns.iloc[0:0],
            benchmark_returns=benchmark_returns.iloc[0:0] if benchmark_returns is not None else None,
            top_n=top_n,
            hold_days=hold_days,
            rebalance_step=rebalance_step,
            cost_rate=cost_rate,
        )

    periods = len(detail)
    total_days = periods * int(hold_days)
    gross_curve = (1.0 + detail["excess_return"].fillna(0.0)).cumprod()
    net_curve = (1.0 + detail["net_excess_return"].fillna(0.0)).cumprod()

    gross_total = float(gross_curve.iloc[-1] - 1.0)
    net_total = float(net_curve.iloc[-1] - 1.0)
    annual_factor = 252.0 / max(1, total_days)
    gross_annual = float((1.0 + gross_total) ** annual_factor - 1.0)
    net_annual = float((1.0 + net_total) ** annual_factor - 1.0)

    hit_rate = float((detail["excess_return"] > 0).mean())
    turnover_mean = float(detail["turnover"].mean())
    max_drawdown = float(_max_drawdown(net_curve.tolist()))

    rank_ic_series = pd.to_numeric(detail["rank_ic"], errors="coerce").dropna()
    rank_ic_mean = float(rank_ic_series.mean()) if not rank_ic_series.empty else 0.0
    if len(rank_ic_series) > 1 and float(rank_ic_series.std(ddof=0)) > 0:
        periods_per_year = 252.0 / max(1, int(hold_days))
        rank_ic_ir = float(rank_ic_series.mean() / rank_ic_series.std(ddof=0) * np.sqrt(periods_per_year))
    else:
        rank_ic_ir = 0.0

    contrib_rows = [
        {"sw_industry": industry, "drawdown_contribution": value}
        for industry, value in sorted(drawdown_contrib.items(), key=lambda item: item[1], reverse=True)
    ]
    contribution = pd.DataFrame(contrib_rows)
    if not contribution.empty:
        total_contrib = float(contribution["drawdown_contribution"].sum())
        contribution["drawdown_share"] = (
            contribution["drawdown_contribution"] / total_contrib if total_contrib > 0 else 0.0
        )
    else:
        contribution = pd.DataFrame(columns=["sw_industry", "drawdown_contribution", "drawdown_share"])

    summary = {
        "periods": int(periods),
        "industry_hit_rate": hit_rate,
        "industry_excess_total_return": gross_total,
        "industry_excess_annual_return": gross_annual,
        "industry_cost_adjusted_total_return": net_total,
        "industry_cost_adjusted_annual_return": net_annual,
        "industry_turnover_mean": turnover_mean,
        "industry_max_drawdown": max_drawdown,
        "industry_rank_ic_mean": rank_ic_mean,
        "industry_rank_ic_ir": rank_ic_ir,
        "trend_gate_mean": float(detail["trend_gate"].mean()) if "trend_gate" in detail.columns else 1.0,
        "crowding_penalty_mean": (
            float(detail["crowding_penalty_mean"].mean()) if "crowding_penalty_mean" in detail.columns else 1.0
        ),
        "exposure_penalty_mean": (
            float(detail["exposure_penalty_mean"].mean()) if "exposure_penalty_mean" in detail.columns else 1.0
        ),
    }
    return detail, contribution, summary
