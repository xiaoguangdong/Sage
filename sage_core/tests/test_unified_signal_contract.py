from __future__ import annotations

import pandas as pd

from sage_core.execution.unified_signal_contract import UNIFIED_SIGNAL_COLUMNS, build_unified_signal_contract


def test_build_unified_signal_contract_includes_stock_industry_trend():
    stock_contract = pd.DataFrame(
        [
            {
                "trade_date": "20260215",
                "strategy_id": "seed_balance_strategy",
                "is_champion": True,
                "signal_name": "stock_score",
                "source": "champion_challenger",
                "ts_code": "000001.SZ",
                "score": 0.9,
                "rank": 1,
                "confidence": 0.8,
                "model_version": "seed_balance_strategy@v1.0.0",
            },
            {
                "trade_date": "20260215",
                "strategy_id": "positive_strategy_v1",
                "is_champion": False,
                "signal_name": "stock_score",
                "source": "champion_challenger",
                "ts_code": "000002.SZ",
                "score": 0.7,
                "rank": 1,
                "confidence": 0.7,
                "model_version": "positive_strategy_v1@v1.0.0",
            },
        ]
    )
    industry_contract = pd.DataFrame(
        [
            {
                "trade_date": "2026-02-15",
                "sw_industry": "电子",
                "signal_name": "policy_score",
                "score": 0.8,
                "confidence": 0.9,
                "direction": 1,
                "source": "policy_signal_pipeline",
                "model_version": "policy_v1",
                "meta": "{}",
            }
        ]
    )
    trend_result = {"state_name": "RISK_ON", "state": 2, "confidence": 0.75, "position_suggestion": 0.8}

    unified = build_unified_signal_contract(
        trade_date="20260215",
        stock_contract=stock_contract,
        industry_contract=industry_contract,
        trend_result=trend_result,
        include_challengers=True,
    )
    assert not unified.empty
    assert list(unified.columns) == UNIFIED_SIGNAL_COLUMNS
    assert set(unified["signal_domain"]) == {"stock", "industry", "trend"}
    assert "000300.SH" in set(unified["entity_id"])


def test_build_unified_signal_contract_can_filter_challengers():
    stock_contract = pd.DataFrame(
        [
            {
                "trade_date": "20260215",
                "strategy_id": "seed_balance_strategy",
                "is_champion": True,
                "signal_name": "stock_score",
                "source": "champion_challenger",
                "ts_code": "000001.SZ",
                "score": 0.9,
                "rank": 1,
                "confidence": 0.8,
                "model_version": "seed_balance_strategy@v1.0.0",
            },
            {
                "trade_date": "20260215",
                "strategy_id": "positive_strategy_v1",
                "is_champion": False,
                "signal_name": "stock_score",
                "source": "champion_challenger",
                "ts_code": "000002.SZ",
                "score": 0.7,
                "rank": 1,
                "confidence": 0.7,
                "model_version": "positive_strategy_v1@v1.0.0",
            },
        ]
    )

    unified = build_unified_signal_contract(
        trade_date="20260215",
        stock_contract=stock_contract,
        industry_contract=None,
        trend_result=None,
        include_challengers=False,
    )
    assert len(unified) == 1
    assert unified.iloc[0]["strategy_id"] == "seed_balance_strategy"
