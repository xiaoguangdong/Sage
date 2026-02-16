#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from sage_core.industry.industry_backtest_eval import (
    build_industry_score_series,
    evaluate_industry_rotation,
    prepare_benchmark_returns,
    prepare_crowding_penalty,
    prepare_industry_returns,
    prepare_trend_gate,
)
from scripts.data._shared.runtime import get_data_path, get_tushare_root


def _resolve_path(raw: str | None, default: Path) -> Path:
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else PROJECT_ROOT / path
    return default


def main() -> None:
    parser = argparse.ArgumentParser(description="行业模型回测评估（命中率/超额/回撤贡献/换手/成本后收益）")
    parser.add_argument("--industry-contract-path", type=str, default=None, help="行业信号契约文件")
    parser.add_argument("--sw-daily-path", type=str, default=None, help="申万行业日线文件")
    parser.add_argument("--sw-l1-map-path", type=str, default=None, help="申万L1映射文件（index_code->industry_name）")
    parser.add_argument("--benchmark-path", type=str, default=None, help="基准指数文件（默认沪深300）")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--top-n", type=int, default=2, help="每次持有行业数")
    parser.add_argument("--hold-days", type=int, default=5, help="每次持有天数（周频默认5）")
    parser.add_argument("--rebalance-step", type=int, default=5, help="调仓步长（交易日）")
    parser.add_argument("--cost-rate", type=float, default=0.005, help="单边交易成本")
    parser.add_argument("--compare-enhanced", action="store_true", help="输出增强方案（趋势门控+拥挤度降温+行业暴露惩罚）对照")
    parser.add_argument("--trend-neutral-multiplier", type=float, default=0.6, help="趋势中性档位仓位乘数")
    parser.add_argument("--trend-risk-off-multiplier", type=float, default=0.2, help="趋势风控档位仓位乘数")
    parser.add_argument("--crowding-threshold-z", type=float, default=1.5, help="拥挤度zscore阈值")
    parser.add_argument("--crowding-penalty-factor", type=float, default=0.8, help="拥挤惩罚系数")
    parser.add_argument("--exposure-penalty-window", type=int, default=12, help="行业暴露惩罚滚动窗口（调仓期数）")
    parser.add_argument("--exposure-penalty-threshold", type=float, default=0.6, help="行业暴露惩罚阈值")
    parser.add_argument("--exposure-penalty-factor", type=float, default=0.85, help="行业暴露惩罚系数")
    args = parser.parse_args()

    tushare_root = get_tushare_root()
    industry_contract_path = _resolve_path(
        args.industry_contract_path,
        get_data_path("signals", "industry", "industry_signal_contract.parquet"),
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
        get_data_path("backtest", "industry", ensure=True),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not industry_contract_path.exists():
        raise FileNotFoundError(f"未找到行业信号契约: {industry_contract_path}")
    if not sw_daily_path.exists():
        raise FileNotFoundError(f"未找到申万行业日线: {sw_daily_path}")

    contract = pd.read_parquet(industry_contract_path)
    scores = build_industry_score_series(contract)

    sw_daily = pd.read_parquet(sw_daily_path)
    sw_l1_map = pd.read_parquet(sw_l1_map_path) if sw_l1_map_path.exists() else pd.DataFrame()
    industry_returns = prepare_industry_returns(sw_daily, sw_l1_map)

    benchmark_returns = pd.DataFrame()
    if benchmark_path.exists():
        benchmark_df = pd.read_parquet(benchmark_path)
        benchmark_returns = prepare_benchmark_returns(benchmark_df)

    detail, contribution, summary = evaluate_industry_rotation(
        industry_scores=scores,
        industry_returns=industry_returns,
        benchmark_returns=benchmark_returns,
        top_n=args.top_n,
        hold_days=args.hold_days,
        rebalance_step=args.rebalance_step,
        cost_rate=args.cost_rate,
    )

    detail_path = output_dir / "industry_backtest_period_metrics.parquet"
    contribution_path = output_dir / "industry_backtest_drawdown_contribution.parquet"
    summary_path = output_dir / "industry_backtest_summary.json"

    detail.to_parquet(detail_path, index=False)
    contribution.to_parquet(contribution_path, index=False)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"行业回测评估完成: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.compare_enhanced:
        trend_gate = pd.DataFrame()
        if benchmark_path.exists():
            benchmark_df = pd.read_parquet(benchmark_path)
            trend_gate = prepare_trend_gate(
                benchmark_df,
                neutral_multiplier=args.trend_neutral_multiplier,
                risk_off_multiplier=args.trend_risk_off_multiplier,
            )
        crowding_penalty = prepare_crowding_penalty(
            sw_daily,
            sw_l1_map,
            z_threshold=args.crowding_threshold_z,
            penalty_factor=args.crowding_penalty_factor,
        )
        enhanced_detail, enhanced_contribution, enhanced_summary = evaluate_industry_rotation(
            industry_scores=scores,
            industry_returns=industry_returns,
            benchmark_returns=benchmark_returns,
            top_n=args.top_n,
            hold_days=args.hold_days,
            rebalance_step=args.rebalance_step,
            cost_rate=args.cost_rate,
            trend_gate=trend_gate,
            crowding_penalty=crowding_penalty,
            enable_exposure_penalty=True,
            exposure_penalty_window=args.exposure_penalty_window,
            exposure_penalty_threshold=args.exposure_penalty_threshold,
            exposure_penalty_factor=args.exposure_penalty_factor,
        )

        enhanced_detail_path = output_dir / "industry_backtest_period_metrics_enhanced.parquet"
        enhanced_contrib_path = output_dir / "industry_backtest_drawdown_contribution_enhanced.parquet"
        enhanced_summary_path = output_dir / "industry_backtest_summary_enhanced.json"
        compare_path = output_dir / "industry_backtest_compare.json"

        enhanced_detail.to_parquet(enhanced_detail_path, index=False)
        enhanced_contribution.to_parquet(enhanced_contrib_path, index=False)
        enhanced_summary_path.write_text(json.dumps(enhanced_summary, ensure_ascii=False, indent=2), encoding="utf-8")

        compare_summary = {
            "baseline": summary,
            "enhanced": enhanced_summary,
            "delta": {
                "industry_excess_annual_return": enhanced_summary["industry_excess_annual_return"] - summary["industry_excess_annual_return"],
                "industry_cost_adjusted_annual_return": enhanced_summary["industry_cost_adjusted_annual_return"] - summary["industry_cost_adjusted_annual_return"],
                "industry_max_drawdown": enhanced_summary["industry_max_drawdown"] - summary["industry_max_drawdown"],
                "industry_hit_rate": enhanced_summary["industry_hit_rate"] - summary["industry_hit_rate"],
                "industry_turnover_mean": enhanced_summary["industry_turnover_mean"] - summary["industry_turnover_mean"],
                "industry_rank_ic_ir": enhanced_summary["industry_rank_ic_ir"] - summary["industry_rank_ic_ir"],
            },
        }
        compare_path.write_text(json.dumps(compare_summary, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"增强方案评估完成: {enhanced_summary_path}")
        print(f"对照结果: {compare_path}")
        print(json.dumps(compare_summary["delta"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
