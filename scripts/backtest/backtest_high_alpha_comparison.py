#!/usr/bin/env python3
"""
高 Alpha 因子对比回测

对比有/无高Alpha因子的选股效果：
1. Baseline: 基础 LGBM 选股器（无高Alpha因子）
2. HighAlpha: 基础 LGBM + 高Alpha因子（moneyflow + margin）
3. HighAlpha+Neutralize: 基础 LGBM + 高Alpha因子 + 因子中性化

输出：
- 各策略的收益、夏普、最大回撤等指标对比
- 因子 IC 分析
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sage_core.backtest.simple_engine import SimpleBacktestEngine
from sage_core.backtest.types import BacktestConfig
from sage_core.stock_selection.stock_selector import SelectionConfig, StockSelector
from scripts.data._shared.runtime import get_data_root, get_tushare_root

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ==================== 数据加载 ====================


def load_daily_data(daily_dir: Path, start_date: str, end_date: str) -> pd.DataFrame:
    """加载日线数据"""
    frames = []
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])

    for year in range(start_year - 1, end_year + 1):
        path = daily_dir / f"daily_{year}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["trade_date"] = df["trade_date"].astype(str)
    return df


def load_daily_basic(path: Path, start_date: str, end_date: str) -> pd.DataFrame:
    """加载每日基本面数据"""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["trade_date"] = df["trade_date"].astype(str)
    return df


def prepare_backtest_data(
    daily_df: pd.DataFrame,
    daily_basic_df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """合并日线和基本面数据"""
    basic_cols = [
        c
        for c in ["ts_code", "trade_date", "pe_ttm", "pb", "turnover_rate", "total_mv", "circ_mv"]
        if c in daily_basic_df.columns
    ]

    df = daily_df.merge(daily_basic_df[basic_cols], on=["ts_code", "trade_date"], how="left")
    return df


def build_returns(daily_df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """构建收益率数据"""
    df = daily_df[(daily_df["trade_date"] >= start_date) & (daily_df["trade_date"] <= end_date)].copy()

    df["ret"] = df.groupby("ts_code")["close"].pct_change()
    df["ret"] = df["ret"].fillna(0)

    cols = ["trade_date", "ts_code", "ret", "close"]
    if "high" in df.columns:
        cols.append("high")
    if "low" in df.columns:
        cols.append("low")
    if "turnover_rate" in df.columns:
        cols.append("turnover_rate")

    return df[cols]


# ==================== 策略构建 ====================


def build_strategy(
    name: str,
    df_train: pd.DataFrame,
    df_backtest: pd.DataFrame,
    tushare_root: Path,
    use_high_alpha: bool = False,
    high_alpha_groups: Optional[List[str]] = None,
    neutralize: bool = False,
) -> Dict[str, Any]:
    """构建并训练策略，返回信号"""
    print(f"\n{'='*60}")
    print(f"策略: {name}")
    print(f"  高Alpha因子: {'启用' if use_high_alpha else '关闭'}")
    if use_high_alpha:
        print(f"  因子组: {high_alpha_groups or '全部可用'}")
    print(f"  因子中性化: {'启用' if neutralize else '关闭'}")
    print(f"{'='*60}")

    cfg = SelectionConfig(
        model_type="lgbm",
        label_horizons=(20, 60, 120),
        label_weights=(0.5, 0.3, 0.2),
        risk_adjusted=True,
        label_neutralized=True,
        ic_filter_enabled=False,
        industry_rank=False,
        exclude_bj=True,
        exclude_st=True,
        # 高Alpha因子配置
        use_high_alpha_features=use_high_alpha,
        high_alpha_groups=high_alpha_groups,
        high_alpha_lookback=20,
        data_root=str(tushare_root),
        # 因子中性化配置
        neutralize_features=neutralize,
        neutralize_method="market_cap",
    )

    selector = StockSelector(cfg)

    # 训练
    print("  训练模型...")
    df_train_copy = df_train.copy()
    df_train_copy["trade_date"] = pd.to_datetime(df_train_copy["trade_date"], format="%Y%m%d", errors="coerce")
    selector.fit(df_train_copy)

    # 预计算特征
    print("  计算回测期特征...")
    df_bt = df_backtest.copy()
    df_bt["trade_date"] = pd.to_datetime(df_bt["trade_date"], format="%Y%m%d", errors="coerce")
    df_prepared = selector.prepare_features(df_bt)

    # 生成信号
    print("  生成信号...")
    trade_dates = sorted(df_prepared["trade_date"].unique())
    all_signals = []

    for td in trade_dates:
        df_day = df_prepared[df_prepared["trade_date"] == td]
        if len(df_day) < 50:
            continue

        try:
            result = selector.predict_prepared(df_day)
            if result.empty:
                continue

            result = result.nlargest(30, "score").copy()
            result["trade_date"] = td.strftime("%Y%m%d") if hasattr(td, "strftime") else str(td)[:8]
            all_signals.append(result[["ts_code", "trade_date", "score"]])
        except Exception:
            continue

    if all_signals:
        signals_df = pd.concat(all_signals, ignore_index=True)
    else:
        signals_df = pd.DataFrame(columns=["ts_code", "trade_date", "score"])

    print(f"  信号数量: {len(signals_df)} 条, 覆盖 {signals_df['trade_date'].nunique()} 个交易日")

    return {
        "name": name,
        "signals": signals_df,
        "selector": selector,
        "config": cfg,
    }


# ==================== 因子 IC 分析 ====================


def compute_factor_ic(signals_df: pd.DataFrame, returns_df: pd.DataFrame) -> Dict[str, float]:
    """计算因子 IC（信息系数）"""
    if signals_df.empty or "score" not in signals_df.columns:
        return {"ic_mean": 0, "ic_std": 0, "ic_ir": 0, "ic_hit_rate": 0}

    merged = signals_df.merge(returns_df[["ts_code", "trade_date", "ret"]], on=["ts_code", "trade_date"], how="inner")

    if merged.empty:
        return {"ic_mean": 0, "ic_std": 0, "ic_ir": 0, "ic_hit_rate": 0}

    # 按日期计算 rank IC
    ic_list = []
    for td, group in merged.groupby("trade_date"):
        if len(group) < 10:
            continue
        ic = group["score"].corr(group["ret"], method="spearman")
        if not pd.isna(ic):
            ic_list.append(ic)

    if not ic_list:
        return {"ic_mean": 0, "ic_std": 0, "ic_ir": 0, "ic_hit_rate": 0}

    ic_series = pd.Series(ic_list)
    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0
    ic_hit_rate = (ic_series > 0).mean()

    return {
        "ic_mean": float(ic_mean),
        "ic_std": float(ic_std),
        "ic_ir": float(ic_ir),
        "ic_hit_rate": float(ic_hit_rate),
    }


# ==================== 主函数 ====================


def main():
    parser = argparse.ArgumentParser(description="高Alpha因子对比回测")
    parser.add_argument("--train-start", type=str, default="20220101", help="训练开始日期")
    parser.add_argument("--train-end", type=str, default="20240901", help="训练结束日期（=回测开始日期）")
    parser.add_argument("--test-end", type=str, default="20260201", help="回测结束日期")
    parser.add_argument("--top-n", type=int, default=10, help="TopN 选股数量")
    parser.add_argument("--data-dir", type=str, default=None, help="Tushare 数据根目录")
    args = parser.parse_args()

    tushare_root = Path(args.data_dir) if args.data_dir else get_tushare_root()
    daily_dir = tushare_root / "daily"
    daily_basic_path = tushare_root / "daily_basic_all.parquet"

    print("=" * 80)
    print("高 Alpha 因子对比回测")
    print("=" * 80)
    print(f"训练区间: {args.train_start} ~ {args.train_end}")
    print(f"回测区间: {args.train_end} ~ {args.test_end}")
    print(f"数据目录: {tushare_root}")

    # 加载数据
    print("\n加载数据...")
    daily_df = load_daily_data(daily_dir, args.train_start, args.test_end)
    daily_basic_df = load_daily_basic(daily_basic_path, args.train_start, args.test_end)

    if daily_df.empty:
        print("错误: 无日线数据")
        return

    print(f"日线数据: {len(daily_df):,} 条")
    print(f"基本面数据: {len(daily_basic_df):,} 条")

    # 合并数据
    df_all = prepare_backtest_data(daily_df, daily_basic_df, args.train_start, args.test_end)

    # 分割训练/回测
    df_train = df_all[(df_all["trade_date"] >= args.train_start) & (df_all["trade_date"] < args.train_end)].copy()
    df_backtest = df_all[(df_all["trade_date"] >= args.train_end) & (df_all["trade_date"] <= args.test_end)].copy()

    print(f"训练数据: {len(df_train):,} 条")
    print(f"回测数据: {len(df_backtest):,} 条")

    # 构建收益率
    returns_df = build_returns(daily_df, args.train_end, args.test_end)
    print(f"收益率数据: {len(returns_df):,} 条")

    # ==================== 构建策略 ====================

    strategies = []

    # 策略1: Baseline（无高Alpha因子）
    s1 = build_strategy(
        name="Baseline (无高Alpha)",
        df_train=df_train,
        df_backtest=df_backtest,
        tushare_root=tushare_root,
        use_high_alpha=False,
    )
    strategies.append(s1)

    # 策略2: 高Alpha因子（moneyflow + margin，跳过 northbound 因为数据缺口）
    s2 = build_strategy(
        name="HighAlpha (moneyflow+margin)",
        df_train=df_train,
        df_backtest=df_backtest,
        tushare_root=tushare_root,
        use_high_alpha=True,
        high_alpha_groups=["moneyflow", "margin"],
    )
    strategies.append(s2)

    # 策略3: 高Alpha因子 + 因子中性化
    s3 = build_strategy(
        name="HighAlpha+Neutralize",
        df_train=df_train,
        df_backtest=df_backtest,
        tushare_root=tushare_root,
        use_high_alpha=True,
        high_alpha_groups=["moneyflow", "margin"],
        neutralize=True,
    )
    strategies.append(s3)

    # ==================== 运行回测 ====================

    print("\n" + "=" * 80)
    print("运行回测...")
    print("=" * 80)

    bt_config = BacktestConfig(
        initial_capital=1_000_000,
        max_positions=args.top_n,
        t_plus_one=True,
        data_delay_days=2,
        weight_method="equal",
    )

    results = []
    for strategy in strategies:
        print(f"\n回测: {strategy['name']}")
        engine = SimpleBacktestEngine(config=bt_config, use_advanced_cost=True)

        signals = strategy["signals"]
        if signals.empty:
            print("  跳过: 无信号")
            results.append({"name": strategy["name"], "metrics": {}, "ic": {}})
            continue

        bt_result = engine.run(signals=signals, returns=returns_df)

        # 计算因子 IC
        ic_metrics = compute_factor_ic(signals, returns_df)

        metrics = bt_result.metrics
        results.append(
            {
                "name": strategy["name"],
                "metrics": metrics,
                "ic": ic_metrics,
                "values": bt_result.values,
            }
        )

        print(f"  累计收益: {metrics.get('total_return', 0):.2%}")
        print(f"  年化收益: {metrics.get('annual_return', 0):.2%}")
        print(f"  夏普比率: {metrics.get('sharpe', 0):.2f}")
        print(f"  最大回撤: {metrics.get('max_drawdown', 0):.2%}")
        print(f"  因子IC均值: {ic_metrics.get('ic_mean', 0):.4f}")
        print(f"  因子ICIR: {ic_metrics.get('ic_ir', 0):.2f}")

    # ==================== 汇总报告 ====================

    print("\n" + "=" * 80)
    print("对比回测结果汇总")
    print("=" * 80)

    header = f"{'策略':<30} {'累计收益':>10} {'年化收益':>10} {'夏普':>8} {'最大回撤':>10} {'IC均值':>8} {'ICIR':>8}"
    print(header)
    print("-" * len(header))

    for r in results:
        m = r["metrics"]
        ic = r["ic"]
        if not m:
            print(f"{r['name']:<30} {'N/A':>10}")
            continue
        print(
            f"{r['name']:<30} "
            f"{m.get('total_return', 0):>9.2%} "
            f"{m.get('annual_return', 0):>9.2%} "
            f"{m.get('sharpe', 0):>7.2f} "
            f"{m.get('max_drawdown', 0):>9.2%} "
            f"{ic.get('ic_mean', 0):>7.4f} "
            f"{ic.get('ic_ir', 0):>7.2f}"
        )

    # 保存结果
    output_dir = get_data_root() / "backtest" / "high_alpha_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_id,
        "train_start": args.train_start,
        "train_end": args.train_end,
        "test_end": args.test_end,
        "top_n": args.top_n,
        "results": [
            {
                "name": r["name"],
                "metrics": {k: float(v) for k, v in r["metrics"].items() if isinstance(v, (int, float))},
                "ic": r["ic"],
            }
            for r in results
        ],
    }

    detail_path = output_dir / f"high_alpha_comparison_{run_id}.json"
    detail_path.write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")

    latest_path = output_dir / "high_alpha_comparison_latest.json"
    latest_path.write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n结果已保存: {output_dir}")
    print(f"详细文件: {detail_path}")

    # 结论
    if len(results) >= 2 and results[0]["metrics"] and results[1]["metrics"]:
        baseline_sharpe = results[0]["metrics"].get("sharpe", 0)
        ha_sharpe = results[1]["metrics"].get("sharpe", 0)
        diff = ha_sharpe - baseline_sharpe
        print(f"\n结论: 高Alpha因子使夏普比率{'提升' if diff > 0 else '下降'} {abs(diff):.2f}")


if __name__ == "__main__":
    main()
