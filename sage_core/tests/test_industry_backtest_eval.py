from __future__ import annotations

import pandas as pd

from sage_core.industry.industry_backtest_eval import (
    build_industry_score_series,
    evaluate_industry_rotation,
    prepare_benchmark_returns,
    prepare_industry_returns,
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
    assert {"periods", "industry_hit_rate", "industry_excess_annual_return", "industry_turnover_mean"}.issubset(summary.keys())
    assert "industry_cost_adjusted_total_return" in summary
    assert "industry_max_drawdown" in summary
    assert {"sw_industry", "drawdown_contribution", "drawdown_share"}.issubset(contribution.columns)
