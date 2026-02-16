#!/usr/bin/env python3
"""
选股模型 Walk-Forward 训练

训练窗口 130 周（约2.5年）→ 测试窗口 26 周（约半年）→ 滚动
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.stock.run_stock_selector_monthly import (
    _build_selector_config,
    _build_training_panel,
    _filter_st_and_new_stocks,
    _load_all_stocks_universe,
    _load_hs300_universe,
    _load_st_stock_list,
    _resolve_tushare_root,
    _save_model_artifact,
)
from sage_core.stock_selection.stock_selector import SelectionConfig, StockSelector


def _find_trade_dates(data_root: Path, start_date: str, end_date: str) -> List[str]:
    """获取交易日列表"""
    daily_dir = data_root / "daily"
    all_dates = set()
    
    for year in range(int(start_date[:4]), int(end_date[:4]) + 1):
        path = daily_dir / f"daily_{year}.parquet"
        if not path.exists():
            continue
        import pyarrow.parquet as pq
        try:
            table = pq.read_table(path, columns=["trade_date"])
            dates = table["trade_date"].to_pylist()
            for d in dates:
                if start_date <= str(d) <= end_date:
                    all_dates.add(str(d))
        except Exception:
            pass
    
    return sorted(all_dates)


def _split_walk_forward(
    trade_dates: List[str],
    train_weeks: int = 130,
    test_weeks: int = 26,
    roll_weeks: int = 26,
) -> List[Tuple[str, str, str, str]]:
    """
    生成 Walk-Forward 时间划分
    
    Returns:
        [(train_start, train_end, test_start, test_end), ...]
    """
    train_days = train_weeks * 5  # 交易日
    test_days = test_weeks * 5
    roll_days = roll_weeks * 5
    
    splits = []
    i = 0
    
    while i + train_days + test_days <= len(trade_dates):
        train_start = trade_dates[i]
        train_end_idx = i + train_days - 1
        test_start_idx = train_end_idx + 1
        test_end_idx = test_start_idx + test_days - 1
        
        if test_end_idx >= len(trade_dates):
            break
        
        train_end = trade_dates[train_end_idx]
        test_start = trade_dates[test_start_idx]
        test_end = trade_dates[test_end_idx]
        
        splits.append((train_start, train_end, test_start, test_end))
        i += roll_days
    
    return splits


def _evaluate_window(
    selector: StockSelector,
    panel: pd.DataFrame,
    test_start: str,
    test_end: str,
    top_n: int = 10,
    label_horizon: int = 20,
) -> Dict[str, float]:
    """评估单个窗口"""
    test_df = panel[(panel["trade_date"] >= test_start) & (panel["trade_date"] <= test_end)]
    
    if test_df.empty:
        return {}
    
    predictions = selector.predict(test_df)
    
    if predictions.empty:
        return {}
    
    # 计算 Rank IC
    predictions["score"] = pd.to_numeric(predictions["score"], errors="coerce")
    
    # 计算未来收益
    panel_sorted = panel.sort_values(["ts_code", "trade_date"])
    panel_sorted["future_ret"] = panel_sorted.groupby("ts_code")["close"].pct_change(label_horizon).shift(-label_horizon)
    
    eval_df = predictions.merge(
        panel_sorted[["ts_code", "trade_date", "future_ret"]],
        on=["ts_code", "trade_date"],
        how="left"
    )
    eval_df = eval_df.dropna(subset=["score", "future_ret"])
    
    if eval_df.empty:
        return {}
    
    # Rank IC
    rank_ic = eval_df.groupby("trade_date").apply(
        lambda g: g["score"].rank(pct=True).corr(g["future_ret"].rank(pct=True))
    ).mean()
    
    # TopN 收益
    top_daily = (
        eval_df.sort_values(["trade_date", "score"], ascending=[True, False])
        .groupby("trade_date", as_index=False)
        .head(top_n)
        .groupby("trade_date")["future_ret"]
        .mean()
    )
    all_daily = eval_df.groupby("trade_date")["future_ret"].mean()
    
    excess_return = (top_daily - all_daily).mean()
    win_rate = (top_daily > all_daily).mean()
    
    return {
        "rank_ic": float(rank_ic) if np.isfinite(rank_ic) else 0.0,
        "top_n_return": float(top_daily.mean()),
        "excess_return": float(excess_return),
        "win_rate": float(win_rate),
        "test_days": int(len(top_daily)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="选股模型 Walk-Forward 训练")
    parser.add_argument("--config", type=str, default="config/app/strategy_governance.yaml")
    parser.add_argument("--start-date", type=str, default="20200101", help="数据起始日期")
    parser.add_argument("--end-date", type=str, default=None, help="数据截止日期，默认最新")
    parser.add_argument("--train-weeks", type=int, default=130, help="训练窗口（周）")
    parser.add_argument("--test-weeks", type=int, default=26, help="测试窗口（周）")
    parser.add_argument("--roll-weeks", type=int, default=26, help="滚动步长（周）")
    parser.add_argument("--top-n", type=int, default=10, help="TopN 评估")
    parser.add_argument("--model-type", type=str, default="lgbm", choices=["rule", "lgbm", "xgb"])
    parser.add_argument("--universe", type=str, default="hs300", choices=["hs300", "all"])
    parser.add_argument("--exclude-st", action="store_true", default=True)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    
    base_cfg = _build_selector_config(config_path)
    selector_cfg = SelectionConfig(
        model_type=args.model_type,
        label_horizons=base_cfg.label_horizons,
        label_weights=base_cfg.label_weights,
        risk_adjusted=base_cfg.risk_adjusted,
        industry_col=base_cfg.industry_col,
        rank_mode=base_cfg.rank_mode,
        industry_rank=base_cfg.industry_rank,
        lgbm_params=base_cfg.lgbm_params,
        xgb_params=base_cfg.xgb_params,
    )

    data_root = _resolve_tushare_root(args.data_dir)
    
    # 确定日期范围
    end_date = args.end_date or datetime.now().strftime("%Y%m%d")
    start_date = args.start_date
    
    # 获取交易日
    trade_dates = _find_trade_dates(data_root, start_date, end_date)
    print(f"交易日范围: {trade_dates[0]} ~ {trade_dates[-1]}，共 {len(trade_dates)} 天")
    
    # 生成时间划分
    splits = _split_walk_forward(
        trade_dates,
        train_weeks=args.train_weeks,
        test_weeks=args.test_weeks,
        roll_weeks=args.roll_weeks,
    )
    print(f"生成 {len(splits)} 个 Walk-Forward 窗口")
    
    # 加载股票池
    if args.universe == "hs300":
        universe = _load_hs300_universe(data_root, end_date)
        print(f"股票池：沪深300，共 {len(universe)} 只")
    else:
        universe = _load_all_stocks_universe(data_root, end_date, exclude_st=args.exclude_st)
        if args.exclude_st:
            st_stocks = _load_st_stock_list(data_root, end_date)
            # 加载日线数据用于新股过滤（需要加载最近几年的数据）
            daily_frames = []
            for year in range(int(end_date[:4]) - 1, int(end_date[:4]) + 1):
                daily_path = data_root / "daily" / f"daily_{year}.parquet"
                if daily_path.exists():
                    daily_frames.append(pd.read_parquet(daily_path))
            if daily_frames:
                daily_df = pd.concat(daily_frames, ignore_index=True)
            else:
                daily_df = pd.DataFrame()
            universe = _filter_st_and_new_stocks(universe, st_stocks, daily_df, min_list_days=60)
        print(f"股票池：全市场，共 {len(universe)} 只")
    
    # Walk-Forward 训练
    results: List[Dict[str, Any]] = []
    
    for i, (train_start, train_end, test_start, test_end) in enumerate(splits):
        print(f"\n{'='*60}")
        print(f"窗口 {i+1}/{len(splits)}")
        print(f"训练: {train_start} ~ {train_end}")
        print(f"测试: {test_start} ~ {test_end}")
        print("="*60)
        
        # 加载训练数据
        train_panel = _build_training_panel(data_root, train_start, train_end, universe)
        if train_panel.empty:
            print("训练数据为空，跳过")
            continue
        
        # 加载测试数据
        full_panel = _build_training_panel(data_root, train_start, test_end, universe)
        
        # 训练
        selector = StockSelector(selector_cfg)
        try:
            selector.fit(train_panel)
        except Exception as e:
            print(f"训练失败: {e}")
            continue
        
        # 评估
        metrics = _evaluate_window(
            selector, full_panel, test_start, test_end,
            top_n=args.top_n,
            label_horizon=int(selector_cfg.label_horizons[0]) if selector_cfg.label_horizons else 20,
        )
        
        if metrics:
            result = {
                "window": i + 1,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "train_rows": len(train_panel),
                "model_type": args.model_type,
                **metrics,
            }
            results.append(result)
            print(f"Rank IC: {metrics['rank_ic']:.4f}")
            print(f"超额收益: {metrics['excess_return']:.2%}")
            print(f"胜率: {metrics['win_rate']:.1%}")
    
    # 汇总
    if results:
        df_results = pd.DataFrame(results)
        
        # 计算平均指标
        avg_metrics = {
            "avg_rank_ic": float(df_results["rank_ic"].mean()),
            "avg_rank_ic_std": float(df_results["rank_ic"].std()),
            "avg_excess_return": float(df_results["excess_return"].mean()),
            "avg_win_rate": float(df_results["win_rate"].mean()),
            "positive_windows": int((df_results["excess_return"] > 0).sum()),
            "total_windows": len(results),
        }
        
        print(f"\n{'='*60}")
        print("Walk-Forward 汇总")
        print("="*60)
        print(f"平均 Rank IC: {avg_metrics['avg_rank_ic']:.4f}")
        print(f"Rank IC 标准差: {avg_metrics['avg_rank_ic_std']:.4f}")
        print(f"平均超额收益: {avg_metrics['avg_excess_return']:.2%}")
        print(f"平均胜率: {avg_metrics['avg_win_rate']:.1%}")
        print(f"正超额窗口: {avg_metrics['positive_windows']}/{avg_metrics['total_windows']}")
        
        # 保存结果
        output_root = Path(args.output_root) if args.output_root else (ROOT / "data" / "backtest" / "walk_forward")
        output_root.mkdir(parents=True, exist_ok=True)
        
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        df_results.to_parquet(output_root / f"walk_forward_results_{run_id}.parquet", index=False)
        
        summary = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "train_weeks": args.train_weeks,
                "test_weeks": args.test_weeks,
                "roll_weeks": args.roll_weeks,
                "model_type": args.model_type,
                "universe": args.universe,
                "exclude_st": args.exclude_st,
            },
            "avg_metrics": avg_metrics,
            "window_results": results,
        }
        
        summary_path = output_root / f"walk_forward_summary_{run_id}.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        
        # 保存最新
        (output_root / "walk_forward_results_latest.parquet").write_bytes(
            df_results.to_parquet(index=False)
        )
        (output_root / "walk_forward_summary_latest.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        
        print(f"\n结果已保存: {output_root}")


if __name__ == "__main__":
    main()
