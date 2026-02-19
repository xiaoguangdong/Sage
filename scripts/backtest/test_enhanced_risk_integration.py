#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试增强风控集成到回测引擎
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from sage_core.backtest.simple_engine import SimpleBacktestEngine
from sage_core.backtest.types import BacktestConfig
from sage_core.portfolio.enhanced_risk_control import RiskControlConfig


def generate_test_data(n_days=252, n_stocks=50):
    """生成测试数据"""
    np.random.seed(42)

    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    dates_str = [d.strftime("%Y%m%d") for d in dates]

    stocks = [f"{i:06d}.SZ" for i in range(1, n_stocks + 1)]
    industries = ["电子", "医药", "消费", "金融", "能源"]

    # 生成收益率数据
    returns_data = []
    for date in dates_str:
        for stock in stocks:
            ret = np.random.normal(0.001, 0.02)  # 日均收益0.1%，波动2%
            close = 10.0 * (1 + np.random.normal(0, 0.1))
            high = close * 1.01
            low = close * 0.99
            returns_data.append(
                {
                    "trade_date": date,
                    "ts_code": stock,
                    "ret": ret,
                    "close": close,
                    "high": high,
                    "low": low,
                }
            )

    returns_df = pd.DataFrame(returns_data)

    # 生成信号数据（月度调仓）
    signals_data = []
    rebalance_dates = dates_str[::20]  # 每20天调仓一次

    for i, date in enumerate(rebalance_dates):
        # 模拟confidence变化
        if i < len(rebalance_dates) // 3:
            confidence = 0.9  # 牛市
        elif i < 2 * len(rebalance_dates) // 3:
            confidence = 0.6  # 震荡
        else:
            confidence = 0.4  # 熊市

        # 随机选择30只股票
        selected_stocks = np.random.choice(stocks, size=30, replace=False)

        for stock in selected_stocks:
            industry = np.random.choice(industries)
            score = np.random.uniform(0, 1)
            signals_data.append(
                {
                    "trade_date": date,
                    "ts_code": stock,
                    "score": score,
                    "industry": industry,
                    "confidence": confidence,
                }
            )

    signals_df = pd.DataFrame(signals_data)

    return signals_df, returns_df


def test_basic_backtest():
    """测试基础回测（无增强风控）"""
    print("=" * 80)
    print("测试1：基础回测（无增强风控）")
    print("=" * 80)

    signals, returns = generate_test_data()

    config = BacktestConfig(
        initial_capital=1000000,
        max_positions=30,
        cost_rate=0.001,
    )

    engine = SimpleBacktestEngine(
        config=config,
        use_advanced_cost=False,
        use_enhanced_risk_control=False,
    )

    result = engine.run(signals, returns)

    print(f"\n初始资金: {config.initial_capital:,.0f}")
    print(f"最终资金: {result.values[-1]:,.0f}")
    print("\n绩效指标:")
    print(f"  总收益率: {result.metrics['total_return']:.2%}")
    print(f"  年化收益: {result.metrics['annual_return']:.2%}")
    print(f"  年化波动: {result.metrics['annual_volatility']:.2%}")
    print(f"  最大回撤: {result.metrics['max_drawdown']:.2%}")
    print(f"  夏普比率: {result.metrics['sharpe']:.2f}")

    # 分析交易
    trades_df = pd.DataFrame(result.trades)
    print("\n交易统计:")
    print(f"  平均持仓数: {trades_df['positions'].mean():.1f}")
    print(f"  平均换手率: {trades_df['turnover'].mean():.2%}")
    print(f"  平均成本: {trades_df['cost'].mean():.4%}")

    return result


def test_enhanced_risk_control_backtest():
    """测试增强风控回测"""
    print("\n" + "=" * 80)
    print("测试2：增强风控回测")
    print("=" * 80)

    signals, returns = generate_test_data()

    config = BacktestConfig(
        initial_capital=1000000,
        max_positions=30,
        cost_rate=0.001,
    )

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
    )

    engine = SimpleBacktestEngine(
        config=config,
        use_advanced_cost=False,
        use_enhanced_risk_control=True,
        risk_control_config=risk_config,
    )

    result = engine.run(signals, returns)

    print(f"\n初始资金: {config.initial_capital:,.0f}")
    print(f"最终资金: {result.values[-1]:,.0f}")
    print("\n绩效指标:")
    print(f"  总收益率: {result.metrics['total_return']:.2%}")
    print(f"  年化收益: {result.metrics['annual_return']:.2%}")
    print(f"  年化波动: {result.metrics['annual_volatility']:.2%}")
    print(f"  最大回撤: {result.metrics['max_drawdown']:.2%}")
    print(f"  夏普比率: {result.metrics['sharpe']:.2f}")
    print(f"  止损事件: {result.metrics.get('stop_loss_events', 0)}次")

    # 分析交易
    trades_df = pd.DataFrame(result.trades)
    print("\n交易统计:")
    print(f"  平均持仓数: {trades_df['positions'].mean():.1f}")
    print(f"  平均换手率: {trades_df['turnover'].mean():.2%}")
    print(f"  平均成本: {trades_df['cost'].mean():.4%}")

    # 分析confidence和止损
    if "confidence" in trades_df.columns:
        print("\n风控统计:")
        print(f"  平均confidence: {trades_df['confidence'].mean():.2f}")
        print(f"  个股止损次数: {trades_df['stop_stocks'].sum()}")
        print(f"  行业止损次数: {trades_df['stop_industries'].sum()}")

    return result


def test_comparison():
    """对比测试"""
    print("\n" + "=" * 80)
    print("测试3：对比分析")
    print("=" * 80)

    signals, returns = generate_test_data()

    config = BacktestConfig(
        initial_capital=1000000,
        max_positions=30,
        cost_rate=0.001,
    )

    # 基础回测
    engine_basic = SimpleBacktestEngine(
        config=config,
        use_advanced_cost=False,
        use_enhanced_risk_control=False,
    )
    result_basic = engine_basic.run(signals, returns)

    # 增强风控回测
    risk_config = RiskControlConfig(
        base_position=0.6,
        max_position=1.0,
        min_position=0.3,
    )
    engine_enhanced = SimpleBacktestEngine(
        config=config,
        use_advanced_cost=False,
        use_enhanced_risk_control=True,
        risk_control_config=risk_config,
    )
    result_enhanced = engine_enhanced.run(signals, returns)

    # 对比
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

    print("\n结论:")
    if result_enhanced.metrics["max_drawdown"] < result_basic.metrics["max_drawdown"]:
        print(
            f"  ✓ 增强风控降低了最大回撤 {(result_basic.metrics['max_drawdown'] - result_enhanced.metrics['max_drawdown']):.2%}"
        )
    if result_enhanced.metrics["sharpe"] > result_basic.metrics["sharpe"]:
        print(f"  ✓ 增强风控提高了夏普比率 {(result_enhanced.metrics['sharpe'] - result_basic.metrics['sharpe']):.2f}")


if __name__ == "__main__":
    test_basic_backtest()
    test_enhanced_risk_control_backtest()
    test_comparison()

    print("\n" + "=" * 80)
    print("✅ 所有测试完成")
    print("=" * 80)
