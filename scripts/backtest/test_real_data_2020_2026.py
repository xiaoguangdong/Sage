#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用2020-2026真实数据测试增强风控
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from datetime import datetime

import pandas as pd

from sage_core.backtest.simple_engine import SimpleBacktestEngine
from sage_core.backtest.types import BacktestConfig
from sage_core.portfolio.enhanced_risk_control import RiskControlConfig


def load_real_data(start_date="20200101", end_date="20261231"):
    """加载真实数据"""
    print("=" * 80)
    print("加载真实数据")
    print("=" * 80)

    # 从分年文件加载日线数据
    data_dir = ROOT / "data" / "tushare" / "daily"

    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return None, None

    # 加载2020-2026年的数据
    print(f"\n加载日线数据: {start_date} - {end_date}")
    dfs = []
    for year in range(2020, 2027):
        year_file = data_dir / f"daily_{year}.parquet"
        if year_file.exists():
            print(f"  加载 {year} 年数据...")
            df = pd.read_parquet(year_file)
            dfs.append(df)
        else:
            print(f"  ⚠ {year} 年数据不存在")

    if not dfs:
        print("❌ 没有找到任何年份的数据")
        return None, None

    daily_df = pd.concat(dfs, ignore_index=True)

    # 过滤日期范围
    daily_df["trade_date"] = daily_df["trade_date"].astype(str)
    daily_df = daily_df[(daily_df["trade_date"] >= start_date) & (daily_df["trade_date"] <= end_date)]

    if daily_df.empty:
        print("❌ 过滤后日线数据为空")
        return None, None

    print(f"\n✓ 加载了 {len(daily_df)} 条日线数据")
    print(f"  日期范围: {daily_df['trade_date'].min()} - {daily_df['trade_date'].max()}")
    print(f"  股票数量: {daily_df['ts_code'].nunique()}")

    # 计算收益率
    daily_df = daily_df.sort_values(["ts_code", "trade_date"])
    daily_df["ret"] = daily_df.groupby("ts_code")["close"].pct_change()
    daily_df = daily_df.dropna(subset=["ret"])

    # 加载行业数据
    print("\n加载行业数据")
    industry_file = ROOT / "data" / "tushare" / "sectors" / "sw_industry_classify.parquet"

    if industry_file.exists():
        industry_df = pd.read_parquet(industry_file)
        # 使用申万一级行业
        if "industry_name" in industry_df.columns:
            industry_map = industry_df[["ts_code", "industry_name"]].copy()
            industry_map = industry_map.rename(columns={"industry_name": "industry"})
        elif "sw_l1_name" in industry_df.columns:
            industry_map = industry_df[["ts_code", "sw_l1_name"]].copy()
            industry_map = industry_map.rename(columns={"sw_l1_name": "industry"})
        else:
            print("⚠ 行业字段不存在，使用默认行业")
            industry_map = pd.DataFrame({"ts_code": daily_df["ts_code"].unique(), "industry": "未知"})

        industry_map = industry_map.drop_duplicates(subset=["ts_code"])
        print(f"✓ 加载了 {len(industry_map)} 只股票的行业信息")
    else:
        print("⚠ 行业数据文件不存在，使用默认行业")
        industry_map = pd.DataFrame({"ts_code": daily_df["ts_code"].unique(), "industry": "未知"})

    return daily_df, industry_map


def generate_simple_signals(daily_df, industry_map, rebalance_freq=20):
    """生成简单的动量信号

    Args:
        daily_df: 日线数据
        industry_map: 行业映射
        rebalance_freq: 调仓频率（天）
    """
    print("\n" + "=" * 80)
    print("生成交易信号")
    print("=" * 80)

    # 计算20日动量
    daily_df = daily_df.sort_values(["ts_code", "trade_date"])
    daily_df["momentum_20"] = daily_df.groupby("ts_code")["close"].pct_change(20)

    # 获取调仓日期
    all_dates = sorted(daily_df["trade_date"].unique())
    rebalance_dates = all_dates[::rebalance_freq]

    print(f"\n调仓频率: 每{rebalance_freq}天")
    print(f"调仓次数: {len(rebalance_dates)}")

    signals = []

    for i, date in enumerate(rebalance_dates):
        # 获取当日数据
        day_data = daily_df[daily_df["trade_date"] == date].copy()

        if day_data.empty:
            continue

        # 过滤：有效动量数据
        day_data = day_data.dropna(subset=["momentum_20"])

        # 过滤：价格 > 5元
        day_data = day_data[day_data["close"] > 5.0]

        # 过滤：成交额 > 1000万
        if "amount" in day_data.columns:
            day_data = day_data[day_data["amount"] > 1000]

        # 按动量排序，选择前50只
        day_data = day_data.sort_values("momentum_20", ascending=False).head(50)

        if day_data.empty:
            continue

        # 合并行业信息
        day_data = day_data.merge(industry_map, on="ts_code", how="left")
        day_data["industry"] = day_data["industry"].fillna("未知")

        # 模拟confidence：根据市场整体表现
        market_ret = day_data["ret"].mean()
        if market_ret > 0.01:
            confidence = 0.9  # 牛市
        elif market_ret > 0:
            confidence = 0.7  # 震荡偏多
        elif market_ret > -0.01:
            confidence = 0.5  # 震荡偏空
        else:
            confidence = 0.3  # 熊市

        for _, row in day_data.iterrows():
            signals.append(
                {
                    "trade_date": date,
                    "ts_code": row["ts_code"],
                    "score": row["momentum_20"],
                    "industry": row["industry"],
                    "confidence": confidence,
                }
            )

    signals_df = pd.DataFrame(signals)
    print(f"\n✓ 生成了 {len(signals_df)} 条信号")
    print(f"  覆盖日期: {len(rebalance_dates)} 个调仓日")
    print(f"  平均每次选股: {len(signals_df) / len(rebalance_dates):.1f} 只")

    return signals_df


def run_backtest_comparison(signals_df, returns_df, industry_map):
    """运行对比回测"""
    print("\n" + "=" * 80)
    print("运行回测对比")
    print("=" * 80)

    config = BacktestConfig(
        initial_capital=1000000,
        max_positions=30,
        cost_rate=0.0003,  # 万三
        data_delay_days=1,  # T+1
        t_plus_one=True,
    )

    # 1. 基础回测
    print("\n【1/2】运行基础回测...")
    engine_basic = SimpleBacktestEngine(
        config=config,
        use_advanced_cost=True,
        use_enhanced_risk_control=False,
    )
    result_basic = engine_basic.run(signals_df, returns_df, industry_map)
    print("✓ 基础回测完成")

    # 2. 增强风控回测
    print("\n【2/2】运行增强风控回测...")
    risk_config = RiskControlConfig(
        base_position=0.6,
        max_position=1.0,
        min_position=0.3,
        atr_stop_loss_multiplier=2.0,
        industry_drawdown_threshold=-0.15,
        enable_atr_stop=True,
        enable_industry_stop=True,
        enable_tiered_drawdown=True,
        enable_daily_shock_stop=True,
        max_single_position=0.10,
        max_industry_exposure=0.30,
    )

    engine_enhanced = SimpleBacktestEngine(
        config=config,
        use_advanced_cost=True,
        use_enhanced_risk_control=True,
        risk_control_config=risk_config,
    )
    result_enhanced = engine_enhanced.run(signals_df, returns_df, industry_map)
    print("✓ 增强风控回测完成")

    return result_basic, result_enhanced


def print_results(result_basic, result_enhanced):
    """打印对比结果"""
    print("\n" + "=" * 80)
    print("回测结果对比")
    print("=" * 80)

    print("\n初始资金: 1,000,000")
    print(f"基础回测最终资金: {result_basic.values[-1]:,.0f}")
    print(f"增强风控最终资金: {result_enhanced.values[-1]:,.0f}")

    print(f"\n{'指标':<20} | {'基础回测':<15} | {'增强风控':<15} | {'差异':<15}")
    print("-" * 80)

    metrics = [
        ("总收益率", "total_return", "%"),
        ("年化收益", "annual_return", "%"),
        ("年化波动", "annual_volatility", "%"),
        ("最大回撤", "max_drawdown", "%"),
        ("夏普比率", "sharpe", ""),
    ]

    for name, key, unit in metrics:
        basic_val = result_basic.metrics[key]
        enhanced_val = result_enhanced.metrics[key]
        diff = enhanced_val - basic_val

        if unit == "%":
            print(f"{name:<20} | {basic_val:<15.2%} | {enhanced_val:<15.2%} | {diff:+.2%}")
        else:
            print(f"{name:<20} | {basic_val:<15.2f} | {enhanced_val:<15.2f} | {diff:+.2f}")

    # 止损统计
    if "stop_loss_events" in result_enhanced.metrics:
        print(f"\n止损事件: {result_enhanced.metrics['stop_loss_events']}次")

    # 交易统计
    trades_basic = pd.DataFrame(result_basic.trades)
    trades_enhanced = pd.DataFrame(result_enhanced.trades)

    print(f"\n{'交易统计':<20} | {'基础回测':<15} | {'增强风控':<15}")
    print("-" * 60)
    print(
        f"{'平均持仓数':<20} | {trades_basic['positions'].mean():<15.1f} | {trades_enhanced['positions'].mean():<15.1f}"
    )
    print(
        f"{'平均换手率':<20} | {trades_basic['turnover'].mean():<15.2%} | {trades_enhanced['turnover'].mean():<15.2%}"
    )
    print(f"{'平均成本':<20} | {trades_basic['cost'].mean():<15.4%} | {trades_enhanced['cost'].mean():<15.4%}")

    if "confidence" in trades_enhanced.columns:
        print(f"{'平均confidence':<20} | {'-':<15} | {trades_enhanced['confidence'].mean():<15.2f}")
    if "stop_stocks" in trades_enhanced.columns:
        print(f"{'个股止损次数':<20} | {'-':<15} | {trades_enhanced['stop_stocks'].sum():<15.0f}")
    if "stop_industries" in trades_enhanced.columns:
        print(f"{'行业止损次数':<20} | {'-':<15} | {trades_enhanced['stop_industries'].sum():<15.0f}")

    # 结论
    print(f"\n{'='*80}")
    print("结论")
    print("=" * 80)

    improvements = []

    if result_enhanced.metrics["max_drawdown"] < result_basic.metrics["max_drawdown"]:
        dd_improve = result_basic.metrics["max_drawdown"] - result_enhanced.metrics["max_drawdown"]
        dd_improve_pct = dd_improve / result_basic.metrics["max_drawdown"] * 100
        improvements.append(f"✓ 最大回撤降低 {dd_improve:.2%} ({dd_improve_pct:.1f}%改善)")

    if result_enhanced.metrics["annual_volatility"] < result_basic.metrics["annual_volatility"]:
        vol_improve = result_basic.metrics["annual_volatility"] - result_enhanced.metrics["annual_volatility"]
        vol_improve_pct = vol_improve / result_basic.metrics["annual_volatility"] * 100
        improvements.append(f"✓ 年化波动降低 {vol_improve:.2%} ({vol_improve_pct:.1f}%改善)")

    if result_enhanced.metrics["sharpe"] > result_basic.metrics["sharpe"]:
        sharpe_improve = result_enhanced.metrics["sharpe"] - result_basic.metrics["sharpe"]
        improvements.append(f"✓ 夏普比率提高 {sharpe_improve:.2f}")

    if result_enhanced.metrics["total_return"] > result_basic.metrics["total_return"]:
        ret_improve = result_enhanced.metrics["total_return"] - result_basic.metrics["total_return"]
        improvements.append(f"✓ 总收益率提高 {ret_improve:.2%}")

    if improvements:
        for imp in improvements:
            print(imp)
    else:
        print("⚠ 增强风控未显著改善指标")


def save_results(result_basic, result_enhanced):
    """保存结果"""
    output_dir = ROOT / "output" / "backtest"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存净值曲线
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    nav_df = pd.DataFrame(
        {
            "date": range(len(result_basic.values)),
            "basic": result_basic.values,
            "enhanced": result_enhanced.values,
        }
    )
    nav_file = output_dir / f"nav_comparison_{timestamp}.csv"
    nav_df.to_csv(nav_file, index=False)
    print(f"\n✓ 净值曲线已保存: {nav_file}")

    # 保存交易记录
    trades_basic = pd.DataFrame(result_basic.trades)
    trades_enhanced = pd.DataFrame(result_enhanced.trades)

    trades_file = output_dir / f"trades_comparison_{timestamp}.csv"
    trades_basic["strategy"] = "basic"
    trades_enhanced["strategy"] = "enhanced"
    trades_all = pd.concat([trades_basic, trades_enhanced])
    trades_all.to_csv(trades_file, index=False)
    print(f"✓ 交易记录已保存: {trades_file}")


if __name__ == "__main__":
    # 1. 加载数据
    daily_df, industry_map = load_real_data(start_date="20200101", end_date="20261231")

    if daily_df is None:
        print("\n❌ 数据加载失败，退出")
        sys.exit(1)

    # 2. 生成信号
    signals_df = generate_simple_signals(daily_df, industry_map, rebalance_freq=20)  # 每月调仓

    # 3. 准备收益率数据
    returns_df = daily_df[["trade_date", "ts_code", "ret", "close", "high", "low"]].copy()

    # 4. 运行回测
    result_basic, result_enhanced = run_backtest_comparison(signals_df, returns_df, industry_map)

    # 5. 打印结果
    print_results(result_basic, result_enhanced)

    # 6. 保存结果
    save_results(result_basic, result_enhanced)

    print("\n" + "=" * 80)
    print("✅ 回测完成")
    print("=" * 80)
