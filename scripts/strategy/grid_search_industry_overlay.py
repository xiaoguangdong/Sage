#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import itertools
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


def _parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _is_pareto_optimal(df: pd.DataFrame, return_col: str, drawdown_col: str) -> pd.Series:
    points = df[[return_col, drawdown_col]].copy().reset_index(drop=True)
    mask = []
    for i in range(len(points)):
        r_i = points.at[i, return_col]
        d_i = points.at[i, drawdown_col]
        dominated = (
            (points[return_col] >= r_i)
            & (points[drawdown_col] <= d_i)
            & ((points[return_col] > r_i) | (points[drawdown_col] < d_i))
        ).any()
        mask.append(not bool(dominated))
    return pd.Series(mask, index=df.index)


def main() -> None:
    parser = argparse.ArgumentParser(description="行业主线增强参数网格搜索（门控/拥挤度/暴露惩罚）")
    parser.add_argument("--industry-contract-path", type=str, default=None)
    parser.add_argument("--sw-daily-path", type=str, default=None)
    parser.add_argument("--sw-l1-map-path", type=str, default=None)
    parser.add_argument("--benchmark-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)

    parser.add_argument("--top-n", type=int, default=2)
    parser.add_argument("--hold-days", type=int, default=5)
    parser.add_argument("--rebalance-step", type=int, default=5)
    parser.add_argument("--cost-rate", type=float, default=0.005)

    parser.add_argument("--neutral-multipliers", type=str, default="0.6,0.7,0.8")
    parser.add_argument("--risk-off-multipliers", type=str, default="0.1,0.2,0.3")
    parser.add_argument("--crowding-z-thresholds", type=str, default="1.2,1.5,1.8")
    parser.add_argument("--crowding-penalty-factors", type=str, default="0.75,0.8,0.85")
    parser.add_argument("--crowding-roll-window", type=int, default=20)
    parser.add_argument("--exposure-windows", type=str, default="8,12")
    parser.add_argument("--exposure-thresholds", type=str, default="0.5,0.6")
    parser.add_argument("--exposure-factors", type=str, default="0.8,0.85")

    parser.add_argument("--drawdown-penalty-weight", type=float, default=0.5, help="目标函数中回撤惩罚权重")
    parser.add_argument("--hit-rate-weight", type=float, default=0.2, help="目标函数中命中率权重")
    parser.add_argument("--max-combos", type=int, default=0, help="最大组合数（0表示全量）")
    parser.add_argument("--progress-every", type=int, default=25, help="每N个组合输出一次进度")
    args = parser.parse_args()

    tushare_root = get_tushare_root()
    industry_contract_path = _resolve_path(
        args.industry_contract_path,
        get_data_path("signals", "industry", "industry_signal_contract.parquet"),
    )
    sw_daily_path = _resolve_path(args.sw_daily_path, tushare_root / "sectors" / "sw_daily_all.parquet")
    sw_l1_map_path = _resolve_path(args.sw_l1_map_path, tushare_root / "sw_industry" / "sw_industry_l1.parquet")
    benchmark_path = _resolve_path(args.benchmark_path, tushare_root / "index" / "index_000300_SH_ohlc.parquet")
    output_dir = _resolve_path(args.output_dir, get_data_path("backtest", "industry", "grid_search", ensure=True))
    output_dir.mkdir(parents=True, exist_ok=True)

    if not industry_contract_path.exists():
        raise FileNotFoundError(f"未找到行业信号契约: {industry_contract_path}")
    if not sw_daily_path.exists():
        raise FileNotFoundError(f"未找到申万行业日线: {sw_daily_path}")

    contract = pd.read_parquet(industry_contract_path)
    sw_daily = pd.read_parquet(sw_daily_path)
    sw_l1_map = pd.read_parquet(sw_l1_map_path) if sw_l1_map_path.exists() else pd.DataFrame()
    benchmark_df = pd.read_parquet(benchmark_path) if benchmark_path.exists() else pd.DataFrame()

    scores = build_industry_score_series(contract)
    industry_returns = prepare_industry_returns(sw_daily, sw_l1_map)
    benchmark_returns = prepare_benchmark_returns(benchmark_df) if not benchmark_df.empty else pd.DataFrame()

    _, _, baseline = evaluate_industry_rotation(
        industry_scores=scores,
        industry_returns=industry_returns,
        benchmark_returns=benchmark_returns,
        top_n=args.top_n,
        hold_days=args.hold_days,
        rebalance_step=args.rebalance_step,
        cost_rate=args.cost_rate,
    )

    neutral_list = _parse_float_list(args.neutral_multipliers)
    risk_off_list = _parse_float_list(args.risk_off_multipliers)
    crowding_z_list = _parse_float_list(args.crowding_z_thresholds)
    crowding_penalty_list = _parse_float_list(args.crowding_penalty_factors)
    exposure_window_list = _parse_int_list(args.exposure_windows)
    exposure_threshold_list = _parse_float_list(args.exposure_thresholds)
    exposure_factor_list = _parse_float_list(args.exposure_factors)

    trend_cache: dict[tuple[float, float], pd.DataFrame] = {}
    if not benchmark_df.empty:
        for neutral, risk_off in itertools.product(neutral_list, risk_off_list):
            if risk_off > neutral:
                continue
            trend_cache[(neutral, risk_off)] = prepare_trend_gate(
                benchmark_df,
                neutral_multiplier=neutral,
                risk_off_multiplier=risk_off,
            )
    else:
        trend_cache[(1.0, 1.0)] = pd.DataFrame()

    crowd_cache: dict[tuple[float, float], pd.DataFrame] = {}
    for z_threshold, penalty in itertools.product(crowding_z_list, crowding_penalty_list):
        crowd_cache[(z_threshold, penalty)] = prepare_crowding_penalty(
            sw_daily,
            sw_l1_map,
            z_threshold=z_threshold,
            penalty_factor=penalty,
            roll_window=args.crowding_roll_window,
        )

    rows: list[dict] = []
    combinations = list(
        itertools.product(
            trend_cache.keys(),
            crowd_cache.keys(),
            exposure_window_list,
            exposure_threshold_list,
            exposure_factor_list,
        )
    )
    if int(args.max_combos) > 0:
        combinations = combinations[: int(args.max_combos)]

    total = len(combinations)
    for idx, (trend_key, crowd_key, exp_window, exp_threshold, exp_factor) in enumerate(combinations, start=1):
        trend_gate = trend_cache[trend_key]
        crowding_penalty = crowd_cache[crowd_key]
        detail, _, summary = evaluate_industry_rotation(
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
            exposure_penalty_window=exp_window,
            exposure_penalty_threshold=exp_threshold,
            exposure_penalty_factor=exp_factor,
        )
        objective = (
            float(summary["industry_cost_adjusted_annual_return"])
            - float(args.drawdown_penalty_weight) * float(summary["industry_max_drawdown"])
            + float(args.hit_rate_weight) * float(summary["industry_hit_rate"])
        )
        rows.append(
            {
                "combo_id": idx,
                "total_combos": total,
                "neutral_multiplier": trend_key[0],
                "risk_off_multiplier": trend_key[1],
                "crowding_z_threshold": crowd_key[0],
                "crowding_penalty_factor": crowd_key[1],
                "exposure_window": exp_window,
                "exposure_threshold": exp_threshold,
                "exposure_factor": exp_factor,
                "objective": objective,
                "periods": summary["periods"],
                "industry_hit_rate": summary["industry_hit_rate"],
                "industry_excess_annual_return": summary["industry_excess_annual_return"],
                "industry_cost_adjusted_annual_return": summary["industry_cost_adjusted_annual_return"],
                "industry_max_drawdown": summary["industry_max_drawdown"],
                "industry_turnover_mean": summary["industry_turnover_mean"],
                "industry_rank_ic_ir": summary["industry_rank_ic_ir"],
                "trend_gate_mean": summary.get("trend_gate_mean", 1.0),
                "crowding_penalty_mean": summary.get("crowding_penalty_mean", 1.0),
                "exposure_penalty_mean": summary.get("exposure_penalty_mean", 1.0),
            }
        )
        if idx % max(1, int(args.progress_every)) == 0 or idx == total:
            print(f"[grid] progress {idx}/{total}")

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        raise RuntimeError("网格搜索无结果")

    result_df["pareto_optimal"] = _is_pareto_optimal(
        result_df,
        return_col="industry_cost_adjusted_annual_return",
        drawdown_col="industry_max_drawdown",
    )
    result_df = result_df.sort_values("objective", ascending=False).reset_index(drop=True)

    result_path = output_dir / "industry_overlay_grid_results.parquet"
    top_path = output_dir / "industry_overlay_grid_top20.json"
    best_path = output_dir / "industry_overlay_grid_best.json"
    pareto_path = output_dir / "industry_overlay_grid_pareto.parquet"
    compare_path = output_dir / "industry_overlay_grid_compare_to_baseline.json"

    result_df.to_parquet(result_path, index=False)
    result_df[result_df["pareto_optimal"]].to_parquet(pareto_path, index=False)

    top20 = result_df.head(20).to_dict(orient="records")
    top_path.write_text(json.dumps(top20, ensure_ascii=False, indent=2), encoding="utf-8")
    best = result_df.iloc[0].to_dict()
    best_path.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")

    baseline_cost_ann = float(baseline["industry_cost_adjusted_annual_return"])
    baseline_mdd = float(baseline["industry_max_drawdown"])
    compare = {
        "baseline": baseline,
        "best": {
            "objective": float(best["objective"]),
            "industry_cost_adjusted_annual_return": float(best["industry_cost_adjusted_annual_return"]),
            "industry_max_drawdown": float(best["industry_max_drawdown"]),
            "industry_hit_rate": float(best["industry_hit_rate"]),
            "params": {
                "neutral_multiplier": float(best["neutral_multiplier"]),
                "risk_off_multiplier": float(best["risk_off_multiplier"]),
                "crowding_z_threshold": float(best["crowding_z_threshold"]),
                "crowding_penalty_factor": float(best["crowding_penalty_factor"]),
                "exposure_window": int(best["exposure_window"]),
                "exposure_threshold": float(best["exposure_threshold"]),
                "exposure_factor": float(best["exposure_factor"]),
            },
        },
        "delta_best_minus_baseline": {
            "industry_cost_adjusted_annual_return": float(best["industry_cost_adjusted_annual_return"])
            - baseline_cost_ann,
            "industry_max_drawdown": float(best["industry_max_drawdown"]) - baseline_mdd,
            "industry_hit_rate": float(best["industry_hit_rate"]) - float(baseline["industry_hit_rate"]),
        },
    }
    compare_path.write_text(json.dumps(compare, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"网格搜索完成: {result_path}")
    print(f"最佳参数: {best_path}")
    print(f"Pareto前沿: {pareto_path}")
    print(json.dumps(compare["best"], ensure_ascii=False, indent=2))
    print(json.dumps(compare["delta_best_minus_baseline"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
