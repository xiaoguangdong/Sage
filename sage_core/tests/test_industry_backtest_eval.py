from __future__ import annotations

import pandas as pd

from sage_core.industry.industry_backtest_eval import (
    blend_industry_scores_with_concept,
    build_industry_score_series,
    evaluate_industry_rotation,
    prepare_benchmark_returns,
    prepare_concept_bias_scores,
    prepare_crowding_penalty,
    prepare_industry_returns,
    prepare_prosperity_scores,
    prepare_trend_dominant_state,
    prepare_trend_gate,
)


def test_prepare_industry_returns_and_benchmark():
    sw_daily = pd.DataFrame(
        [
            {"trade_date": "2026-01-01", "ts_code": "801010.SI", "pct_change": 1.0},
            {"trade_date": "2026-01-02", "ts_code": "801010.SI", "pct_change": 2.0},
        ]
    )
    sw_l1_map = pd.DataFrame([{"index_code": "801010.SI", "industry_name": "农林牧渔"}])
    ind_ret = prepare_industry_returns(sw_daily, sw_l1_map)
    assert {"trade_date", "sw_industry", "industry_return"}.issubset(ind_ret.columns)
    assert ind_ret["sw_industry"].iloc[0] == "农林牧渔"

    index_df = pd.DataFrame(
        [
            {"trade_date": "2026-01-01", "pct_chg": 0.5},
            {"trade_date": "2026-01-02", "pct_chg": -0.3},
        ]
    )
    bench = prepare_benchmark_returns(index_df)
    assert {"trade_date", "benchmark_return"}.issubset(bench.columns)


def test_evaluate_industry_rotation_outputs_required_metrics():
    contract = pd.DataFrame(
        [
            {"trade_date": "2026-01-01", "sw_industry": "电子", "score": 0.9, "confidence": 0.8},
            {"trade_date": "2026-01-01", "sw_industry": "煤炭", "score": 0.2, "confidence": 0.8},
            {"trade_date": "2026-01-06", "sw_industry": "电子", "score": 0.7, "confidence": 0.8},
            {"trade_date": "2026-01-06", "sw_industry": "煤炭", "score": 0.6, "confidence": 0.8},
            {"trade_date": "2026-01-13", "sw_industry": "电子", "score": 0.3, "confidence": 0.8},
            {"trade_date": "2026-01-13", "sw_industry": "煤炭", "score": 0.8, "confidence": 0.8},
        ]
    )
    scores = build_industry_score_series(contract)

    dates = pd.date_range("2026-01-01", periods=20, freq="B")
    rows = []
    for date in dates:
        date_str = date.strftime("%Y%m%d")
        rows.append({"trade_date": date_str, "sw_industry": "电子", "industry_return": 0.002})
        rows.append({"trade_date": date_str, "sw_industry": "煤炭", "industry_return": 0.001})
    industry_returns = pd.DataFrame(rows)
    benchmark_returns = pd.DataFrame(
        {"trade_date": [d.strftime("%Y%m%d") for d in dates], "benchmark_return": [0.0015] * len(dates)}
    )

    detail, contribution, summary = evaluate_industry_rotation(
        industry_scores=scores,
        industry_returns=industry_returns,
        benchmark_returns=benchmark_returns,
        top_n=1,
        hold_days=2,
        rebalance_step=3,
        cost_rate=0.001,
    )
    assert not detail.empty
    assert {"trade_date", "excess_return", "turnover", "net_excess_return", "rank_ic"}.issubset(detail.columns)
    assert {"periods", "industry_hit_rate", "industry_excess_annual_return", "industry_turnover_mean"}.issubset(
        summary.keys()
    )
    assert "industry_cost_adjusted_total_return" in summary
    assert "industry_max_drawdown" in summary
    assert {"sw_industry", "drawdown_contribution", "drawdown_share"}.issubset(contribution.columns)


def test_prepare_trend_gate_and_crowding_penalty():
    index_df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2025-01-01", periods=80, freq="B").strftime("%Y%m%d"),
            "close": [100 + i * 0.2 for i in range(80)],
        }
    )
    trend_gate = prepare_trend_gate(index_df, neutral_multiplier=0.6, risk_off_multiplier=0.2)
    assert {"trade_date", "trend_gate", "trend_state"}.issubset(trend_gate.columns)
    assert trend_gate["trend_gate"].between(0.0, 1.0).all()

    sw_daily = pd.DataFrame(
        {
            "trade_date": pd.date_range("2025-01-01", periods=40, freq="B").strftime("%Y%m%d").tolist() * 2,
            "ts_code": ["801010.SI"] * 40 + ["801020.SI"] * 40,
            "amount": [1000] * 39 + [20000] + [800] * 40,
            "pct_change": [0.5] * 39 + [5.0] + [0.3] * 40,
        }
    )
    sw_l1_map = pd.DataFrame(
        [
            {"index_code": "801010.SI", "industry_name": "行业A"},
            {"index_code": "801020.SI", "industry_name": "行业B"},
        ]
    )
    crowd = prepare_crowding_penalty(sw_daily, sw_l1_map, z_threshold=1.0, penalty_factor=0.8, roll_window=10)
    assert {"trade_date", "sw_industry", "crowding_penalty"}.issubset(crowd.columns)
    assert (crowd["crowding_penalty"] < 1.0).any()


def test_prepare_prosperity_scores():
    dates = pd.date_range("2025-01-01", periods=40, freq="B")
    rows = []
    for idx, date in enumerate(dates):
        date_str = date.strftime("%Y%m%d")
        rows.append(
            {"trade_date": date_str, "ts_code": "801010.SI", "pct_change": 0.5 + idx * 0.01, "amount": 1000 + idx * 5}
        )
        rows.append(
            {"trade_date": date_str, "ts_code": "801020.SI", "pct_change": -0.2 + idx * 0.005, "amount": 800 + idx * 3}
        )
    sw_daily = pd.DataFrame(rows)
    sw_l1_map = pd.DataFrame(
        [
            {"index_code": "801010.SI", "industry_name": "行业A"},
            {"index_code": "801020.SI", "industry_name": "行业B"},
        ]
    )
    scores = prepare_prosperity_scores(
        sw_daily,
        sw_l1_map,
        momentum_window=10,
        amount_window=10,
        volatility_window=10,
    )
    assert not scores.empty
    assert {"trade_date", "sw_industry", "score", "confidence", "rank"}.issubset(scores.columns)
    assert scores["score"].between(0.0, 1.0).all()
    assert scores["confidence"].between(0.0, 1.0).all()
    assert scores["trade_date"].nunique() >= 20


def test_evaluate_industry_rotation_with_penalties():
    contract = pd.DataFrame(
        [
            {"trade_date": "2026-01-01", "sw_industry": "行业A", "score": 0.9, "confidence": 0.9},
            {"trade_date": "2026-01-01", "sw_industry": "行业B", "score": 0.2, "confidence": 0.9},
            {"trade_date": "2026-01-08", "sw_industry": "行业A", "score": 0.9, "confidence": 0.9},
            {"trade_date": "2026-01-08", "sw_industry": "行业B", "score": 0.2, "confidence": 0.9},
            {"trade_date": "2026-01-15", "sw_industry": "行业A", "score": 0.9, "confidence": 0.9},
            {"trade_date": "2026-01-15", "sw_industry": "行业B", "score": 0.2, "confidence": 0.9},
        ]
    )
    scores = build_industry_score_series(contract)

    dates = pd.date_range("2026-01-01", periods=25, freq="B")
    ret_rows = []
    for date in dates:
        date_str = date.strftime("%Y%m%d")
        ret_rows.append({"trade_date": date_str, "sw_industry": "行业A", "industry_return": 0.01})
        ret_rows.append({"trade_date": date_str, "sw_industry": "行业B", "industry_return": 0.002})
    industry_returns = pd.DataFrame(ret_rows)
    benchmark_returns = pd.DataFrame(
        {"trade_date": [d.strftime("%Y%m%d") for d in dates], "benchmark_return": [0.004] * len(dates)}
    )

    baseline_detail, _, baseline_summary = evaluate_industry_rotation(
        industry_scores=scores,
        industry_returns=industry_returns,
        benchmark_returns=benchmark_returns,
        top_n=1,
        hold_days=2,
        rebalance_step=5,
        cost_rate=0.001,
    )
    assert not baseline_detail.empty

    trend_gate = pd.DataFrame({"trade_date": baseline_detail["trade_date"], "trend_gate": 0.5})
    crowd_penalty = pd.DataFrame(
        {
            "trade_date": baseline_detail["trade_date"],
            "sw_industry": "行业A",
            "crowding_penalty": 0.7,
        }
    )
    enhanced_detail, _, enhanced_summary = evaluate_industry_rotation(
        industry_scores=scores,
        industry_returns=industry_returns,
        benchmark_returns=benchmark_returns,
        top_n=1,
        hold_days=2,
        rebalance_step=5,
        cost_rate=0.001,
        trend_gate=trend_gate,
        crowding_penalty=crowd_penalty,
        enable_exposure_penalty=True,
        exposure_penalty_window=2,
        exposure_penalty_threshold=0.5,
        exposure_penalty_factor=0.8,
    )
    assert not enhanced_detail.empty
    assert (
        enhanced_summary["industry_cost_adjusted_total_return"]
        < baseline_summary["industry_cost_adjusted_total_return"]
    )


def test_prepare_trend_dominant_state_and_blend_scores():
    trend_gate = pd.DataFrame(
        [
            {"trade_date": "2026-01-05", "trend_state": "RISK_ON"},
            {"trade_date": "2026-01-06", "trend_state": "RISK_ON"},
            {"trade_date": "2026-01-07", "trend_state": "RISK_OFF"},
        ]
    )
    dominant = prepare_trend_dominant_state(trend_gate, lookback_days=2, dominance_threshold=0.6)
    assert {"trade_date", "dominant_state", "dominant_ratio"}.issubset(dominant.columns)
    state_map = dict(zip(dominant["trade_date"], dominant["dominant_state"]))
    assert state_map["20260106"] == "RISK_ON"
    assert state_map["20260107"] == "NEUTRAL"

    concept = prepare_concept_bias_scores(
        pd.DataFrame(
            [
                {
                    "trade_date": "2026-01-06",
                    "sw_industry": "电子",
                    "concept_bias_strength": 1.0,
                    "concept_signal_confidence": 1.0,
                },
                {
                    "trade_date": "2026-01-06",
                    "sw_industry": "煤炭",
                    "concept_bias_strength": -1.0,
                    "concept_signal_confidence": 1.0,
                },
            ]
        )
    )
    scores = pd.DataFrame(
        [
            {"trade_date": "20260106", "sw_industry": "电子", "score": 0.60, "confidence": 0.8},
            {"trade_date": "20260106", "sw_industry": "煤炭", "score": 0.55, "confidence": 0.8},
            {"trade_date": "20260107", "sw_industry": "电子", "score": 0.60, "confidence": 0.8},
            {"trade_date": "20260107", "sw_industry": "煤炭", "score": 0.55, "confidence": 0.8},
        ]
    )
    blended = blend_industry_scores_with_concept(
        industry_scores=scores,
        concept_bias=concept,
        dominant_state=dominant,
        risk_on_concept_weight=0.6,
        neutral_concept_weight=0.3,
        risk_off_concept_weight=0.1,
        concept_max_stale_days=10,
    )
    day_on = blended[blended["trade_date"] == "20260106"].set_index("sw_industry")
    day_neutral = blended[blended["trade_date"] == "20260107"].set_index("sw_industry")
    assert day_on.loc["电子", "score"] > day_neutral.loc["电子", "score"]
    assert day_on.loc["煤炭", "score"] < day_neutral.loc["煤炭", "score"]
