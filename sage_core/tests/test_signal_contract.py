from __future__ import annotations

import pandas as pd

from sage_core.execution.signal_contract import (
    apply_industry_overlay,
    build_stock_industry_map_from_features,
    build_stock_signal_contract,
    select_champion_signals,
)


def _sample_signal(trade_date: str, ts_codes: list[str], base_score: float) -> pd.DataFrame:
    rows = []
    for idx, code in enumerate(ts_codes, start=1):
        rows.append(
            {
                "trade_date": trade_date,
                "ts_code": code,
                "score": base_score - idx * 0.01,
                "rank": idx,
                "confidence": 1.0 - idx * 0.05,
                "model_version": "demo@v1",
            }
        )
    return pd.DataFrame(rows)


def test_build_stock_signal_contract_and_select_champion():
    trade_date = "20260215"
    champion = _sample_signal(trade_date, ["000001.SZ", "000002.SZ"], base_score=0.90)
    challenger = {
        "positive_strategy_v1": _sample_signal(trade_date, ["000003.SZ"], base_score=0.88),
    }

    contract = build_stock_signal_contract(
        trade_date=trade_date,
        champion_id="seed_balance_strategy",
        champion_signals=champion,
        challenger_signals=challenger,
        include_challengers=True,
    )
    assert {"strategy_id", "is_champion", "signal_name", "source"}.issubset(contract.columns)
    assert len(contract) == 3

    selected = select_champion_signals(contract, champion_id="seed_balance_strategy")
    assert len(selected) == 2
    assert set(selected["ts_code"]) == {"000001.SZ", "000002.SZ"}


def test_apply_industry_overlay_changes_score():
    trade_date = "20260215"
    champion = _sample_signal(trade_date, ["000001.SZ", "000002.SZ"], base_score=1.00)
    champion["strategy_id"] = "seed_balance_strategy"
    champion["is_champion"] = True
    champion["signal_name"] = "stock_score"
    champion["source"] = "champion_challenger"

    features = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "industry_l1": "电子", "trade_date": "20260214"},
            {"ts_code": "000002.SZ", "industry_l1": "煤炭", "trade_date": "20260214"},
        ]
    )
    stock_industry_map = build_stock_industry_map_from_features(features)

    industry_snapshot = pd.DataFrame(
        [
            {"trade_date": trade_date, "sw_industry": "电子", "signal_name": "policy_score", "score": 0.8, "confidence": 0.9},
            {"trade_date": trade_date, "sw_industry": "电子", "signal_name": "concept_bias", "score": 0.7, "confidence": 0.8},
            {"trade_date": trade_date, "sw_industry": "电子", "signal_name": "northbound_ratio", "score": 0.6, "confidence": 0.7},
            {"trade_date": trade_date, "sw_industry": "煤炭", "signal_name": "policy_score", "score": -0.6, "confidence": 0.8},
            {"trade_date": trade_date, "sw_industry": "煤炭", "signal_name": "concept_bias", "score": -0.4, "confidence": 0.7},
            {"trade_date": trade_date, "sw_industry": "煤炭", "signal_name": "northbound_ratio", "score": -0.5, "confidence": 0.9},
        ]
    )

    out = apply_industry_overlay(
        stock_signals=champion,
        industry_snapshot=industry_snapshot,
        stock_industry_map=stock_industry_map,
        overlay_strength=0.2,
    )
    assert {"industry_l1", "industry_overlay", "score_raw", "score_final"}.issubset(out.columns)

    elec_score = out.loc[out["ts_code"] == "000001.SZ", "score_final"].iloc[0]
    coal_score = out.loc[out["ts_code"] == "000002.SZ", "score_final"].iloc[0]
    assert elec_score > out.loc[out["ts_code"] == "000001.SZ", "score_raw"].iloc[0]
    assert coal_score < out.loc[out["ts_code"] == "000002.SZ", "score_raw"].iloc[0]


def test_apply_industry_overlay_supports_zero_one_score_with_direction():
    trade_date = "20260215"
    champion = _sample_signal(trade_date, ["000001.SZ", "000002.SZ"], base_score=1.00)
    champion["strategy_id"] = "seed_balance_strategy"
    champion["is_champion"] = True
    champion["signal_name"] = "stock_score"
    champion["source"] = "champion_challenger"

    stock_industry_map = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "industry_l1": "电子"},
            {"ts_code": "000002.SZ", "industry_l1": "煤炭"},
        ]
    )

    industry_snapshot = pd.DataFrame(
        [
            {"trade_date": trade_date, "sw_industry": "电子", "signal_name": "policy_score", "score": 0.9, "confidence": 1.0, "direction": 1},
            {"trade_date": trade_date, "sw_industry": "煤炭", "signal_name": "policy_score", "score": 0.1, "confidence": 1.0, "direction": -1},
        ]
    )
    out = apply_industry_overlay(
        stock_signals=champion,
        industry_snapshot=industry_snapshot,
        stock_industry_map=stock_industry_map,
        signal_weights={"policy_score": 1.0},
        overlay_strength=0.2,
    )
    elec_score = out.loc[out["ts_code"] == "000001.SZ", "score_final"].iloc[0]
    coal_score = out.loc[out["ts_code"] == "000002.SZ", "score_final"].iloc[0]
    assert elec_score > out.loc[out["ts_code"] == "000001.SZ", "score_raw"].iloc[0]
    assert coal_score < out.loc[out["ts_code"] == "000002.SZ", "score_raw"].iloc[0]
