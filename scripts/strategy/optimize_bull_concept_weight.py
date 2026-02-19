#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from sage_core.industry.industry_backtest_eval import (
    blend_industry_scores_with_concept,
    evaluate_industry_rotation,
    prepare_benchmark_returns,
    prepare_concept_bias_scores,
    prepare_industry_returns,
    prepare_prosperity_scores,
    prepare_trend_dominant_state,
    prepare_trend_gate,
)
from scripts.data._shared.runtime import get_data_path, get_tushare_root


def _resolve_path(raw: str | None, default: Path) -> Path:
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else PROJECT_ROOT / path
    return default


def _parse_float_list(raw: str) -> List[float]:
    values = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            continue
        values.append(float(text))
    if not values:
        raise ValueError("weights 为空")
    return values


def _segment_metrics(
    detail: pd.DataFrame, dominant_state: pd.DataFrame, state_name: str, hold_days: int
) -> Dict[str, float]:
    if detail.empty:
        return {
            "periods": 0,
            "net_excess_mean": 0.0,
            "net_excess_total": 0.0,
            "net_excess_annual": 0.0,
            "net_excess_hit_rate": 0.0,
        }
    merged = detail[["trade_date", "net_excess_return"]].merge(
        dominant_state[["trade_date", "dominant_state"]],
        on="trade_date",
        how="left",
    )
    selected = merged[merged["dominant_state"] == state_name].copy()
    if selected.empty:
        return {
            "periods": 0,
            "net_excess_mean": 0.0,
            "net_excess_total": 0.0,
            "net_excess_annual": 0.0,
            "net_excess_hit_rate": 0.0,
        }

    series = selected["net_excess_return"].fillna(0.0)
    total = float((1.0 + series).prod() - 1.0)
    total_days = max(1, int(len(series) * int(hold_days)))
    annual = float((1.0 + total) ** (252.0 / total_days) - 1.0)
    return {
        "periods": int(len(series)),
        "net_excess_mean": float(series.mean()),
        "net_excess_total": total,
        "net_excess_annual": annual,
        "net_excess_hit_rate": float((series > 0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="仅优化牛市概念权重（NEUTRAL/RISK_OFF 固定）")
    parser.add_argument("--industry-contract-path", type=str, default=None)
    parser.add_argument("--concept-bias-path", type=str, default=None)
    parser.add_argument("--sw-daily-path", type=str, default=None)
    parser.add_argument("--sw-l1-map-path", type=str, default=None)
    parser.add_argument("--benchmark-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--risk-on-weights", type=str, default="0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--neutral-weight", type=float, default=0.3)
    parser.add_argument("--risk-off-weight", type=float, default=0.1)
    parser.add_argument("--concept-max-stale-days", type=int, default=7)
    parser.add_argument("--trend-neutral-multiplier", type=float, default=0.6)
    parser.add_argument("--trend-risk-off-multiplier", type=float, default=0.2)
    parser.add_argument("--trend-dominance-lookback-days", type=int, default=5)
    parser.add_argument("--trend-dominance-threshold", type=float, default=0.6)
    parser.add_argument("--prosperity-momentum-window", type=int, default=20)
    parser.add_argument("--prosperity-amount-window", type=int, default=20)
    parser.add_argument("--prosperity-volatility-window", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=2)
    parser.add_argument("--hold-days", type=int, default=5)
    parser.add_argument("--rebalance-step", type=int, default=5)
    parser.add_argument("--cost-rate", type=float, default=0.005)
    parser.add_argument("--max-full-annual-drop", type=float, default=0.02, help="相对景气基线允许的最大年化下降")
    parser.add_argument("--max-full-mdd-increase", type=float, default=0.02, help="相对景气基线允许的最大回撤上升")
    parser.add_argument("--min-bull-annual-improve", type=float, default=0.0, help="相对景气基线要求的最小牛市年化改善")
    args = parser.parse_args()

    risk_on_weights = _parse_float_list(args.risk_on_weights)

    tushare_root = get_tushare_root()
    industry_contract_path = _resolve_path(
        args.industry_contract_path,
        get_data_path("signals", "industry", "industry_signal_contract.parquet"),
    )
    concept_bias_path = _resolve_path(
        args.concept_bias_path,
        get_data_path("signals", "industry", "industry_concept_bias.parquet"),
    )
    sw_daily_path = _resolve_path(
        args.sw_daily_path,
        tushare_root / "sectors" / "sw_daily_all.parquet",
    )
    sw_l1_map_path = _resolve_path(
        args.sw_l1_map_path,
        tushare_root / "sw_industry" / "sw_industry_l1.parquet",
    )
    benchmark_path = _resolve_path(
        args.benchmark_path,
        tushare_root / "index" / "index_000300_SH_ohlc.parquet",
    )
    output_dir = _resolve_path(
        args.output_dir,
        get_data_path("backtest", "industry", "bull_weight_grid", ensure=True),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not sw_daily_path.exists():
        raise FileNotFoundError(f"未找到申万行业日线: {sw_daily_path}")
    if not sw_l1_map_path.exists():
        raise FileNotFoundError(f"未找到申万L1映射: {sw_l1_map_path}")
    if not benchmark_path.exists():
        raise FileNotFoundError(f"未找到基准指数: {benchmark_path}")
    if not concept_bias_path.exists():
        raise FileNotFoundError(f"未找到概念偏置信号: {concept_bias_path}")
    if not industry_contract_path.exists():
        raise FileNotFoundError(f"未找到行业信号契约: {industry_contract_path}")

    sw_daily = pd.read_parquet(sw_daily_path)
    sw_l1_map = pd.read_parquet(sw_l1_map_path)
    benchmark_df = pd.read_parquet(benchmark_path)
    concept_bias = prepare_concept_bias_scores(pd.read_parquet(concept_bias_path))

    industry_returns = prepare_industry_returns(sw_daily, sw_l1_map)
    benchmark_returns = prepare_benchmark_returns(benchmark_df)
    trend_gate = prepare_trend_gate(
        benchmark_df,
        neutral_multiplier=args.trend_neutral_multiplier,
        risk_off_multiplier=args.trend_risk_off_multiplier,
    )
    dominant_state = prepare_trend_dominant_state(
        trend_gate,
        lookback_days=args.trend_dominance_lookback_days,
        dominance_threshold=args.trend_dominance_threshold,
    )
    prosperity_scores = prepare_prosperity_scores(
        sw_daily,
        sw_l1_map,
        momentum_window=args.prosperity_momentum_window,
        amount_window=args.prosperity_amount_window,
        volatility_window=args.prosperity_volatility_window,
    )

    base_detail, _, base_summary = evaluate_industry_rotation(
        industry_scores=prosperity_scores,
        industry_returns=industry_returns,
        benchmark_returns=benchmark_returns,
        top_n=args.top_n,
        hold_days=args.hold_days,
        rebalance_step=args.rebalance_step,
        cost_rate=args.cost_rate,
    )
    base_bull = _segment_metrics(base_detail, dominant_state, "RISK_ON", args.hold_days)

    rows = []
    for risk_on_weight in risk_on_weights:
        blended = blend_industry_scores_with_concept(
            prosperity_scores,
            concept_bias,
            dominant_state,
            risk_on_concept_weight=risk_on_weight,
            neutral_concept_weight=args.neutral_weight,
            risk_off_concept_weight=args.risk_off_weight,
            concept_max_stale_days=args.concept_max_stale_days,
        )
        detail, _, summary = evaluate_industry_rotation(
            industry_scores=blended,
            industry_returns=industry_returns,
            benchmark_returns=benchmark_returns,
            top_n=args.top_n,
            hold_days=args.hold_days,
            rebalance_step=args.rebalance_step,
            cost_rate=args.cost_rate,
        )
        bull = _segment_metrics(detail, dominant_state, "RISK_ON", args.hold_days)
        rows.append(
            {
                "risk_on_weight": float(risk_on_weight),
                "neutral_weight": float(args.neutral_weight),
                "risk_off_weight": float(args.risk_off_weight),
                "full_periods": int(summary["periods"]),
                "full_cost_adjusted_annual_return": float(summary["industry_cost_adjusted_annual_return"]),
                "full_max_drawdown": float(summary["industry_max_drawdown"]),
                "full_hit_rate": float(summary["industry_hit_rate"]),
                "bull_periods": int(bull["periods"]),
                "bull_net_excess_annual_return": float(bull["net_excess_annual"]),
                "bull_net_excess_total_return": float(bull["net_excess_total"]),
                "bull_net_excess_hit_rate": float(bull["net_excess_hit_rate"]),
                "delta_full_annual_vs_prosperity": float(
                    summary["industry_cost_adjusted_annual_return"]
                    - base_summary["industry_cost_adjusted_annual_return"]
                ),
                "delta_full_mdd_vs_prosperity": float(
                    summary["industry_max_drawdown"] - base_summary["industry_max_drawdown"]
                ),
                "delta_bull_annual_vs_prosperity": float(bull["net_excess_annual"] - base_bull["net_excess_annual"]),
                "delta_bull_total_vs_prosperity": float(bull["net_excess_total"] - base_bull["net_excess_total"]),
            }
        )

    results = (
        pd.DataFrame(rows)
        .sort_values(
            ["delta_bull_annual_vs_prosperity", "delta_full_annual_vs_prosperity"],
            ascending=False,
        )
        .reset_index(drop=True)
    )
    if results.empty:
        feasible = pd.DataFrame()
    else:
        feasible = results[
            (results["delta_full_annual_vs_prosperity"] >= -float(args.max_full_annual_drop))
            & (results["delta_full_mdd_vs_prosperity"] <= float(args.max_full_mdd_increase))
            & (results["delta_bull_annual_vs_prosperity"] >= float(args.min_bull_annual_improve))
        ].copy()
        feasible = feasible.sort_values(
            ["delta_bull_annual_vs_prosperity", "delta_full_annual_vs_prosperity", "delta_full_mdd_vs_prosperity"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
    best = (
        feasible.iloc[0].to_dict() if not feasible.empty else (results.iloc[0].to_dict() if not results.empty else {})
    )

    report = {
        "baseline_prosperity_only": {
            "full_cost_adjusted_annual_return": float(base_summary["industry_cost_adjusted_annual_return"]),
            "full_max_drawdown": float(base_summary["industry_max_drawdown"]),
            "full_hit_rate": float(base_summary["industry_hit_rate"]),
            "bull_metrics": base_bull,
        },
        "params": {
            "risk_on_weights": risk_on_weights,
            "neutral_weight": float(args.neutral_weight),
            "risk_off_weight": float(args.risk_off_weight),
            "concept_max_stale_days": int(args.concept_max_stale_days),
            "top_n": int(args.top_n),
            "hold_days": int(args.hold_days),
            "rebalance_step": int(args.rebalance_step),
            "cost_rate": float(args.cost_rate),
            "max_full_annual_drop": float(args.max_full_annual_drop),
            "max_full_mdd_increase": float(args.max_full_mdd_increase),
            "min_bull_annual_improve": float(args.min_bull_annual_improve),
        },
        "feasible_count": int(len(feasible)),
        "feasible_ratio": float(len(feasible) / max(1, len(results))) if not results.empty else 0.0,
        "selection_mode": "feasible_first",
        "best": best,
    }

    results_path = output_dir / "bull_concept_weight_grid_results.parquet"
    top10_path = output_dir / "bull_concept_weight_grid_top10.json"
    report_path = output_dir / "bull_concept_weight_grid_report.json"
    results.to_parquet(results_path, index=False)
    top10_path.write_text(
        json.dumps(results.head(10).to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    feasible_top10_path = output_dir / "bull_concept_weight_grid_feasible_top10.json"
    feasible_top10_path.write_text(
        json.dumps(feasible.head(10).to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"已输出: {results_path}")
    print(f"已输出: {top10_path}")
    print(f"已输出: {feasible_top10_path}")
    print(f"已输出: {report_path}")
    if best:
        print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
