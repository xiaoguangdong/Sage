"""
Walk-Forward 评估测试 — 验证选股模型真实表现
"""
import os
import sys
import logging

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sage_core.stock_selection.stock_selector import SelectionConfig
from sage_core.stock_selection.walk_forward import WalkForwardConfig, WalkForwardEvaluator

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "tushare")


def load_stock_data(years: list) -> pd.DataFrame:
    frames = []
    for y in years:
        path = os.path.join(DATA_ROOT, "daily", f"daily_{y}.parquet")
        if os.path.exists(path):
            frames.append(pd.read_parquet(path))
    df = pd.concat(frames, ignore_index=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    if "turnover" not in df.columns and "vol" in df.columns:
        df["turnover"] = df["vol"]
    print(f"股票数据: {len(df)} 条, {df['ts_code'].nunique()} 只股票")
    return df


def load_daily_basic() -> pd.DataFrame:
    path = os.path.join(DATA_ROOT, "daily_basic_all.parquet")
    df = pd.read_parquet(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    return df


def load_industry_mapping() -> pd.DataFrame:
    path = os.path.join(DATA_ROOT, "sw_industry", "sw_industry_l1.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()


def merge_data(df_stock, df_basic, df_industry):
    basic_cols = ["ts_code", "trade_date", "pe_ttm", "pb", "turnover_rate", "total_mv", "circ_mv"]
    available = [c for c in basic_cols if c in df_basic.columns]
    df = df_stock.merge(df_basic[available], on=["ts_code", "trade_date"], how="left")
    if not df_industry.empty and "ts_code" in df_industry.columns:
        ind_col = "industry_name" if "industry_name" in df_industry.columns else "industry_l1"
        if ind_col in df_industry.columns:
            df = df.merge(
                df_industry[["ts_code", ind_col]].drop_duplicates("ts_code"),
                on="ts_code", how="left",
            )
            df.rename(columns={ind_col: "industry_l1"}, inplace=True)
    return df


def run_test():
    print("=" * 60)
    print("Walk-Forward 选股模型评估")
    print("=" * 60)

    # 加载数据
    df_stock = load_stock_data([2020, 2021, 2022, 2023])
    df_basic = load_daily_basic()
    df_industry = load_industry_mapping()
    df = merge_data(df_stock, df_basic, df_industry)

    # 选股配置（新默认: 10/20天标签）
    sel_cfg = SelectionConfig(
        model_type="lgbm",
        label_neutralized=True,
        market_neutralized=False,
        ic_filter_enabled=True,
        ic_threshold=0.02,
        ic_ir_threshold=0.3,
        industry_col="industry_l1",
    )
    print(f"\n标签周期: {sel_cfg.label_horizons}, 权重: {sel_cfg.label_weights}")

    # Walk-Forward 配置
    wf_cfg = WalkForwardConfig(
        train_days=504,     # 2年训练
        val_days=63,        # 1季度验证
        step_days=63,       # 1季度步进
        purge_days=30,      # 30天 purge（>= 20天标签 × 1.5）
        embargo_days=5,
        n_quantiles=5,
        forward_days=20,    # 20天前瞻收益评估
    )

    # 运行
    evaluator = WalkForwardEvaluator(wf_cfg, sel_cfg)
    result = evaluator.run(df)

    # 输出结果
    print("\n" + "=" * 60)
    print("Walk-Forward 评估结果")
    print("=" * 60)

    # 窗口详情
    print(f"\n窗口数: {len(result.window_details)}")
    for w in result.window_details:
        if w.get("status") == "ok":
            print(f"  Window {w['window_id']}: "
                  f"{w['train_start']}~{w['train_end']} → {w['val_start']}~{w['val_end']} "
                  f"({w['n_features']} features)")
        else:
            print(f"  Window {w['window_id']}: {w['status']}")

    # 分组收益
    s = result.summary
    if "error" not in s:
        print(f"\n分组回测 (Quintile Analysis):")
        print(f"  评估日期数: {s['n_dates']}")
        print(f"  每日平均股票数: {s['n_stocks_per_date']:.0f}")
        print(f"\n  各组平均 {wf_cfg.forward_days} 日收益:")
        for q, ret in s["quantile_mean_returns"].items():
            bar = "█" * int(abs(ret) * 1000)
            sign = "+" if ret > 0 else ""
            print(f"    {q}: {sign}{ret*100:.3f}%  {bar}")
        print(f"\n  多空收益 (Q1-Q5): {s['long_short_return']*100:.3f}%")
        print(f"  单调性: {s['monotonic_ratio']*100:.0f}%")
        print(f"  Rank IC: {s['rank_ic']:.4f}")
        print(f"  Rank IC_IR: {s['rank_ic_ir']:.4f}")
        print(f"  IC 胜率: {s['ic_hit_rate']*100:.1f}%")

        # 判断模型质量
        print("\n" + "-" * 40)
        if s["rank_ic"] > 0.03 and s["ic_hit_rate"] > 0.55:
            print("✓ 模型有效：IC > 3%, 胜率 > 55%")
        elif s["rank_ic"] > 0.02:
            print("△ 模型边际有效：IC 在 2-3% 之间，需进一步优化")
        else:
            print("✗ 模型无效：IC < 2%，需要重新设计")

        if s["monotonic_ratio"] >= 0.75:
            print("✓ 分组单调性良好")
        else:
            print("△ 分组单调性不足，模型区分度有限")
    else:
        print(f"\n评估失败: {s['error']}")

    print("\n✓ 测试完成！")


if __name__ == "__main__":
    run_test()
