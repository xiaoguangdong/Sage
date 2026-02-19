from __future__ import annotations

import json
from typing import Any, Dict, Optional

import pandas as pd

UNIFIED_SIGNAL_COLUMNS = [
    "trade_date",
    "signal_domain",
    "entity_type",
    "entity_id",
    "signal_name",
    "score",
    "confidence",
    "direction",
    "rank",
    "source",
    "model_version",
    "strategy_id",
    "is_champion",
    "meta",
]


def _normalize_trade_date(value: Any) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.notna(ts):
        return ts.strftime("%Y%m%d")
    return str(value)


def _to_json_meta(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "{}"
    return json.dumps(value, ensure_ascii=False)


def from_stock_signal_contract(
    stock_contract: Optional[pd.DataFrame], include_challengers: bool = True
) -> pd.DataFrame:
    if stock_contract is None or stock_contract.empty:
        return pd.DataFrame(columns=UNIFIED_SIGNAL_COLUMNS)

    required = {
        "trade_date",
        "ts_code",
        "signal_name",
        "score",
        "confidence",
        "rank",
        "source",
        "model_version",
        "strategy_id",
        "is_champion",
    }
    missing = required - set(stock_contract.columns)
    if missing:
        raise ValueError(f"stock_signal_contract缺少字段: {sorted(missing)}")

    data = stock_contract.copy()
    if not include_challengers:
        data = data[data["is_champion"]].copy()
    if data.empty:
        return pd.DataFrame(columns=UNIFIED_SIGNAL_COLUMNS)

    out = pd.DataFrame()
    out["trade_date"] = data["trade_date"].map(_normalize_trade_date)
    out["signal_domain"] = "stock"
    out["entity_type"] = "stock"
    out["entity_id"] = data["ts_code"].astype(str)
    out["signal_name"] = data["signal_name"].astype(str)
    out["score"] = pd.to_numeric(data["score"], errors="coerce").fillna(0.0)
    out["confidence"] = pd.to_numeric(data["confidence"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out["direction"] = 1
    out["rank"] = pd.to_numeric(data["rank"], errors="coerce").fillna(9999).astype(int)
    out["source"] = data["source"].astype(str)
    out["model_version"] = data["model_version"].astype(str)
    out["strategy_id"] = data["strategy_id"].astype(str)
    out["is_champion"] = data["is_champion"].astype(bool)
    out["meta"] = "{}"
    return (
        out[UNIFIED_SIGNAL_COLUMNS]
        .sort_values(["trade_date", "is_champion", "strategy_id", "rank"], ascending=[True, False, True, True])
        .reset_index(drop=True)
    )


def from_industry_signal_contract(industry_contract: Optional[pd.DataFrame]) -> pd.DataFrame:
    if industry_contract is None or industry_contract.empty:
        return pd.DataFrame(columns=UNIFIED_SIGNAL_COLUMNS)

    required = {
        "trade_date",
        "sw_industry",
        "signal_name",
        "score",
        "confidence",
        "source",
        "model_version",
    }
    missing = required - set(industry_contract.columns)
    if missing:
        raise ValueError(f"industry_signal_contract缺少字段: {sorted(missing)}")

    data = industry_contract.copy()
    data["trade_date"] = data["trade_date"].map(_normalize_trade_date)
    data["score"] = pd.to_numeric(data["score"], errors="coerce").fillna(0.5)
    data["confidence"] = pd.to_numeric(data["confidence"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    if "direction" in data.columns:
        data["direction"] = pd.to_numeric(data["direction"], errors="coerce").fillna(0).clip(-1, 1).astype(int)
    else:
        data["direction"] = 0
        data.loc[data["score"] > 0.55, "direction"] = 1
        data.loc[data["score"] < 0.45, "direction"] = -1

    data["rank"] = (
        data.groupby(["trade_date", "signal_name"])["score"]
        .rank(ascending=False, method="first")
        .fillna(9999)
        .astype(int)
    )

    out = pd.DataFrame()
    out["trade_date"] = data["trade_date"]
    out["signal_domain"] = "industry"
    out["entity_type"] = "industry"
    out["entity_id"] = data["sw_industry"].astype(str)
    out["signal_name"] = data["signal_name"].astype(str)
    out["score"] = data["score"]
    out["confidence"] = data["confidence"]
    out["direction"] = data["direction"]
    out["rank"] = data["rank"]
    out["source"] = data["source"].astype(str)
    out["model_version"] = data["model_version"].astype(str)
    out["strategy_id"] = None
    out["is_champion"] = None
    if "meta" in data.columns:
        out["meta"] = data["meta"].apply(_to_json_meta)
    else:
        out["meta"] = "{}"
    return out[UNIFIED_SIGNAL_COLUMNS].sort_values(["trade_date", "signal_name", "rank"]).reset_index(drop=True)


def from_trend_result(
    trade_date: str,
    trend_result: Optional[Dict[str, Any]],
    source: str = "trend_model",
    model_version: str = "trend_rule_v1",
) -> pd.DataFrame:
    if not trend_result:
        return pd.DataFrame(columns=UNIFIED_SIGNAL_COLUMNS)

    state = trend_result.get("state_name", trend_result.get("state", "NEUTRAL"))
    if isinstance(state, (int, float)):
        state_map = {0: "RISK_OFF", 1: "NEUTRAL", 2: "RISK_ON"}
        state_name = state_map.get(int(state), "NEUTRAL")
    else:
        text = str(state).upper()
        if text in {"RISK_ON", "BULL"}:
            state_name = "RISK_ON"
        elif text in {"RISK_OFF", "BEAR"}:
            state_name = "RISK_OFF"
        else:
            state_name = "NEUTRAL"

    score_map = {"RISK_OFF": 0.0, "NEUTRAL": 0.5, "RISK_ON": 1.0}
    score = score_map[state_name]
    direction = 1 if score > 0.55 else (-1 if score < 0.45 else 0)
    confidence = float(trend_result.get("confidence", trend_result.get("label_main_confidence", 0.5)))
    confidence = min(1.0, max(0.0, confidence))
    meta = {
        "state_name": state_name,
        "state_raw": trend_result.get("state"),
        "position_suggestion": trend_result.get("position_suggestion"),
    }

    out = pd.DataFrame(
        [
            {
                "trade_date": _normalize_trade_date(trade_date),
                "signal_domain": "trend",
                "entity_type": "market",
                "entity_id": "000300.SH",
                "signal_name": "trend_state",
                "score": score,
                "confidence": confidence,
                "direction": direction,
                "rank": 1,
                "source": source,
                "model_version": model_version,
                "strategy_id": None,
                "is_champion": None,
                "meta": _to_json_meta(meta),
            }
        ]
    )
    return out[UNIFIED_SIGNAL_COLUMNS]


def build_unified_signal_contract(
    *,
    trade_date: str,
    stock_contract: Optional[pd.DataFrame] = None,
    industry_contract: Optional[pd.DataFrame] = None,
    trend_result: Optional[Dict[str, Any]] = None,
    include_challengers: bool = True,
) -> pd.DataFrame:
    frames = [
        from_stock_signal_contract(stock_contract, include_challengers=include_challengers),
        from_industry_signal_contract(industry_contract),
        from_trend_result(trade_date, trend_result),
    ]
    frames = [item for item in frames if item is not None and not item.empty]
    if not frames:
        return pd.DataFrame(columns=UNIFIED_SIGNAL_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out["trade_date"] = out["trade_date"].map(_normalize_trade_date)
    return (
        out[UNIFIED_SIGNAL_COLUMNS]
        .sort_values(
            ["trade_date", "signal_domain", "signal_name", "rank"],
            ascending=[True, True, True, True],
        )
        .reset_index(drop=True)
    )
