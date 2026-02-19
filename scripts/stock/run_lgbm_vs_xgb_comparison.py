#!/usr/bin/env python3
"""
LGBM vs XGBoost 系统对比训练

基于 Walk-Forward 框架，在相同数据、相同特征、相同标签下对比两个模型。
训练窗口 130 周（约2.5年）→ 测试窗口 26 周（约半年）→ 滚动
数据范围：2020-2026
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sage_core.stock_selection.stock_selector import SelectionConfig, StockSelector
from scripts.data._shared.runtime import get_data_root
from scripts.stock.run_stock_selector_monthly import (
    _build_selector_config,
    _build_training_panel,
    _load_hs300_universe,
    _resolve_tushare_root,
)
from scripts.stock.run_stock_walk_forward import _evaluate_window, _find_trade_dates, _split_walk_forward


def _run_single_model(
    model_type: str,
    base_cfg: SelectionConfig,
    train_panel: pd.DataFrame,
    full_panel: pd.DataFrame,
    test_start: str,
    test_end: str,
    top_n: int,
) -> Dict[str, Any]:
    """训练并评估单个模型，返回指标字典"""
    cfg = replace(base_cfg, model_type=model_type)
    selector = StockSelector(cfg)

    try:
        selector.fit(train_panel)
    except Exception as e:
        return {"status": "failed", "error": str(e)}

    metrics = _evaluate_window(
        selector,
        full_panel,
        test_start,
        test_end,
        top_n=top_n,
        label_horizon=int(cfg.label_horizons[0]) if cfg.label_horizons else 20,
    )

    if not metrics:
        return {"status": "no_data"}

    return {
        "status": "ok",
        "feature_count": len(selector.feature_cols or []),
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LGBM vs XGBoost 系统对比训练")
    parser.add_argument("--config", type=str, default="config/app/strategy_governance.yaml")
    parser.add_argument("--start-date", type=str, default="20200101")
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--train-weeks", type=int, default=130)
    parser.add_argument("--test-weeks", type=int, default=26)
    parser.add_argument("--roll-weeks", type=int, default=26)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    base_cfg = _build_selector_config(config_path)

    data_root = _resolve_tushare_root(args.data_dir)
    end_date = args.end_date or datetime.now().strftime("%Y%m%d")

    trade_dates = _find_trade_dates(data_root, args.start_date, end_date)
    if not trade_dates:
        raise ValueError("无交易日数据")
    print(f"交易日: {trade_dates[0]} ~ {trade_dates[-1]}，共 {len(trade_dates)} 天")

    splits = _split_walk_forward(
        trade_dates,
        train_weeks=args.train_weeks,
        test_weeks=args.test_weeks,
        roll_weeks=args.roll_weeks,
    )
    print(f"Walk-Forward 窗口: {len(splits)} 个")

    universe = _load_hs300_universe(data_root, end_date)
    print(f"股票池: 沪深300，{len(universe)} 只")

    model_types = ["lgbm", "xgb"]
    all_results: Dict[str, List[Dict[str, Any]]] = {m: [] for m in model_types}

    for i, (train_start, train_end, test_start, test_end) in enumerate(splits):
        print(f"\n{'='*60}")
        print(f"窗口 {i+1}/{len(splits)}: 训练 {train_start}~{train_end} | 测试 {test_start}~{test_end}")
        print("=" * 60)

        train_panel = _build_training_panel(data_root, train_start, train_end, universe)
        if train_panel.empty:
            print("训练数据为空，跳过")
            continue

        full_panel = _build_training_panel(data_root, train_start, test_end, universe)

        for model_type in model_types:
            print(f"\n  [{model_type.upper()}] 训练中...")
            result = _run_single_model(
                model_type=model_type,
                base_cfg=base_cfg,
                train_panel=train_panel,
                full_panel=full_panel,
                test_start=test_start,
                test_end=test_end,
                top_n=args.top_n,
            )

            if result["status"] == "ok":
                row = {
                    "window": i + 1,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "train_rows": len(train_panel),
                    "model_type": model_type,
                    **{k: v for k, v in result.items() if k != "status"},
                }
                all_results[model_type].append(row)
                print(
                    f"  [{model_type.upper()}] Rank IC={result['rank_ic']:.4f}, "
                    f"超额={result['excess_return']:.2%}, 胜率={result['win_rate']:.1%}"
                )
            else:
                print(f"  [{model_type.upper()}] {result['status']}: {result.get('error', '')}")

    # 汇总对比
    print(f"\n{'='*60}")
    print("LGBM vs XGBoost 对比汇总")
    print("=" * 60)

    comparison = {}
    for model_type in model_types:
        rows = all_results[model_type]
        if not rows:
            print(f"\n{model_type.upper()}: 无有效结果")
            continue

        df = pd.DataFrame(rows)
        stats = {
            "windows": len(df),
            "avg_rank_ic": float(df["rank_ic"].mean()),
            "rank_ic_std": float(df["rank_ic"].std()),
            "rank_ic_ir": float(df["rank_ic"].mean() / df["rank_ic"].std()) if df["rank_ic"].std() > 0 else 0.0,
            "avg_excess_return": float(df["excess_return"].mean()),
            "avg_win_rate": float(df["win_rate"].mean()),
            "positive_windows": int((df["excess_return"] > 0).sum()),
            "avg_top_n_return": float(df["top_n_return"].mean()),
        }
        comparison[model_type] = stats

        print(f"\n{model_type.upper()}:")
        print(f"  平均 Rank IC:    {stats['avg_rank_ic']:.4f} ± {stats['rank_ic_std']:.4f}")
        print(f"  Rank IC IR:      {stats['rank_ic_ir']:.4f}")
        print(f"  平均超额收益:    {stats['avg_excess_return']:.2%}")
        print(f"  平均胜率:        {stats['avg_win_rate']:.1%}")
        print(f"  正超额窗口:      {stats['positive_windows']}/{stats['windows']}")
        print(f"  平均 TopN 收益:  {stats['avg_top_n_return']:.2%}")

    # 胜负判定
    if len(comparison) == 2:
        lgbm_s = comparison["lgbm"]
        xgb_s = comparison["xgb"]

        print(f"\n{'─'*60}")
        print("逐维度对比:")

        dims = [
            ("Rank IC", "avg_rank_ic", True),
            ("IC IR", "rank_ic_ir", True),
            ("超额收益", "avg_excess_return", True),
            ("胜率", "avg_win_rate", True),
        ]
        lgbm_wins = 0
        for name, key, higher_better in dims:
            l_val, x_val = lgbm_s[key], xgb_s[key]
            winner = "LGBM" if (l_val > x_val) == higher_better else "XGBoost"
            if winner == "LGBM":
                lgbm_wins += 1
            print(f"  {name:12s}: LGBM={l_val:.4f}  XGB={x_val:.4f}  → {winner}")

        overall = "LGBM" if lgbm_wins > len(dims) / 2 else "XGBoost"
        print(f"\n综合胜出: {overall} ({lgbm_wins}/{len(dims)} 维度领先)")

        # 逐窗口配对对比
        if all_results["lgbm"] and all_results["xgb"]:
            lgbm_df = pd.DataFrame(all_results["lgbm"]).set_index("window")
            xgb_df = pd.DataFrame(all_results["xgb"]).set_index("window")
            common = lgbm_df.index.intersection(xgb_df.index)
            if len(common) > 0:
                paired_diff = lgbm_df.loc[common, "excess_return"] - xgb_df.loc[common, "excess_return"]
                lgbm_better = int((paired_diff > 0).sum())
                print(f"\n逐窗口配对: LGBM胜 {lgbm_better}/{len(common)} 个窗口")
                print(f"  配对超额差均值: {paired_diff.mean():.2%}")

    # 保存结果
    output_root = Path(args.output_root) if args.output_root else (get_data_root() / "backtest" / "lgbm_vs_xgb")
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 合并所有窗口结果
    all_rows = []
    for rows in all_results.values():
        all_rows.extend(rows)
    if all_rows:
        df_all = pd.DataFrame(all_rows)
        df_all.to_parquet(output_root / f"lgbm_vs_xgb_{run_id}.parquet", index=False)

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_id,
        "data_range": f"{args.start_date} ~ {end_date}",
        "config": {
            "train_weeks": args.train_weeks,
            "test_weeks": args.test_weeks,
            "roll_weeks": args.roll_weeks,
            "top_n": args.top_n,
        },
        "comparison": comparison,
        "window_results": {m: rows for m, rows in all_results.items()},
    }
    summary_path = output_root / f"lgbm_vs_xgb_{run_id}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "lgbm_vs_xgb_latest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n结果已保存: {output_root}")


if __name__ == "__main__":
    main()
