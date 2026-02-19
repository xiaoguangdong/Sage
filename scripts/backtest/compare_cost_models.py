#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
对比不同成本模型的回测结果

比较三种成本模型：
1. 简单成本率（0.05%单边）
2. 保守成本模型（完整模型）
3. 激进成本模型（完整模型，乐观估计）
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from sage_core.backtest.cost_model import create_aggressive_cost_model, create_default_cost_model
from sage_core.backtest.simple_engine import SimpleBacktestEngine
from sage_core.backtest.types import BacktestConfig


def generate_mock_signals(n_days: int = 252, n_stocks: int = 50) -> pd.DataFrame:
    """生成模拟信号数据"""
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    dates = [d.strftime("%Y%m%d") for d in dates]

    signals = []
    for date in dates:
        # 每天随机选10只股票
        stocks = [f"{i:06d}.SZ" for i in np.random.choice(range(1, n_stocks + 1), 10, replace=False)]
        for stock in stocks:
            signals.append(
                {
                    "trade_date": date,
                    "ts_code": stock,
                    "score": np.random.random(),
                }
            )

    return pd.DataFrame(signals)


def generate_mock_returns(signals: pd.DataFrame) -> pd.DataFrame:
    """生成模拟收益率数据"""
    dates = signals["trade_date"].unique()
    stocks = signals["ts_code"].unique()

    returns = []
    for date in dates:
        for stock in stocks:
            # 模拟日收益率（均值0.05%，标准差2%）
            ret = np.random.normal(0.0005, 0.02)
            returns.append(
                {
                    "trade_date": date,
                    "ts_code": stock,
                    "ret": ret,
                    "volatility": abs(np.random.normal(0.02, 0.005)),  # 波动率
                    "amount": np.random.uniform(1_000_000, 50_000_000),  # 成交额
                    "close": np.random.uniform(5, 50),  # 收盘价
                }
            )

    return pd.DataFrame(returns)


def run_comparison():
    """运行成本模型对比"""
    print("=" * 80)
    print("成本模型对比回测")
    print("=" * 80)

    # 生成模拟数据
    print("\n生成模拟数据...")
    signals = generate_mock_signals(n_days=252, n_stocks=50)
    returns = generate_mock_returns(signals)

    print(f"  信号数据: {len(signals)} 条")
    print(f"  收益数据: {len(returns)} 条")
    print(f"  交易日数: {signals['trade_date'].nunique()} 天")

    # 配置
    config = BacktestConfig(
        initial_capital=10_000_000,
        cost_rate=0.0005,  # 简单模型使用0.05%
        max_positions=10,
        t_plus_one=True,
        data_delay_days=2,
    )

    # 1. 简单成本模型
    print("\n" + "=" * 80)
    print("【模型1】简单成本率（0.05%单边）")
    print("=" * 80)

    engine1 = SimpleBacktestEngine(config=config, use_advanced_cost=False)
    result1 = engine1.run(signals, returns)

    print(f"  总收益: {result1.metrics['total_return']:.2%}")
    print(f"  年化收益: {result1.metrics['annual_return']:.2%}")
    print(f"  年化波动: {result1.metrics['annual_volatility']:.2%}")
    print(f"  最大回撤: {result1.metrics['max_drawdown']:.2%}")
    print(f"  夏普比率: {result1.metrics['sharpe']:.2f}")

    # 计算平均成本
    avg_cost = np.mean([t["cost"] for t in result1.trades if t["turnover"] > 0])
    avg_turnover = np.mean([t["turnover"] for t in result1.trades])
    print(f"  平均换手: {avg_turnover:.2%}")
    print(f"  平均成本: {avg_cost:.4%}")

    # 2. 保守成本模型
    print("\n" + "=" * 80)
    print("【模型2】保守成本模型（完整模型）")
    print("=" * 80)

    cost_model_default = create_default_cost_model()
    print("  成本模型配置:")
    print(f"    固定滑点: {cost_model_default.config.fixed_slippage_bps} bp")
    print(f"    波动率滑点系数: {cost_model_default.config.volatility_slippage_factor}")
    print(f"    市场冲击系数: {cost_model_default.config.market_impact_factor}")

    engine2 = SimpleBacktestEngine(
        config=config,
        cost_model=cost_model_default,
        use_advanced_cost=True,
    )
    result2 = engine2.run(signals, returns)

    print(f"  总收益: {result2.metrics['total_return']:.2%}")
    print(f"  年化收益: {result2.metrics['annual_return']:.2%}")
    print(f"  年化波动: {result2.metrics['annual_volatility']:.2%}")
    print(f"  最大回撤: {result2.metrics['max_drawdown']:.2%}")
    print(f"  夏普比率: {result2.metrics['sharpe']:.2f}")

    avg_cost = np.mean([t["cost"] for t in result2.trades if t["turnover"] > 0])
    avg_turnover = np.mean([t["turnover"] for t in result2.trades])
    print(f"  平均换手: {avg_turnover:.2%}")
    print(f"  平均成本: {avg_cost:.4%}")

    # 3. 激进成本模型
    print("\n" + "=" * 80)
    print("【模型3】激进成本模型（完整模型，乐观估计）")
    print("=" * 80)

    engine3 = SimpleBacktestEngine(
        config=config,
        cost_model=create_aggressive_cost_model(),
        use_advanced_cost=True,
    )
    result3 = engine3.run(signals, returns)

    print(f"  总收益: {result3.metrics['total_return']:.2%}")
    print(f"  年化收益: {result3.metrics['annual_return']:.2%}")
    print(f"  年化波动: {result3.metrics['annual_volatility']:.2%}")
    print(f"  最大回撤: {result3.metrics['max_drawdown']:.2%}")
    print(f"  夏普比率: {result3.metrics['sharpe']:.2f}")

    avg_cost = np.mean([t["cost"] for t in result3.trades if t["turnover"] > 0])
    avg_turnover = np.mean([t["turnover"] for t in result3.trades])
    print(f"  平均换手: {avg_turnover:.2%}")
    print(f"  平均成本: {avg_cost:.4%}")

    # 对比总结
    print("\n" + "=" * 80)
    print("【对比总结】")
    print("=" * 80)

    print(f"\n{'指标':<20} | {'简单模型':>12} | {'保守模型':>12} | {'激进模型':>12}")
    print("-" * 80)
    print(
        f"{'总收益':<20} | {result1.metrics['total_return']:>11.2%} | {result2.metrics['total_return']:>11.2%} | {result3.metrics['total_return']:>11.2%}"
    )
    print(
        f"{'年化收益':<20} | {result1.metrics['annual_return']:>11.2%} | {result2.metrics['annual_return']:>11.2%} | {result3.metrics['annual_return']:>11.2%}"
    )
    print(
        f"{'夏普比率':<20} | {result1.metrics['sharpe']:>12.2f} | {result2.metrics['sharpe']:>12.2f} | {result3.metrics['sharpe']:>12.2f}"
    )
    print(
        f"{'最大回撤':<20} | {result1.metrics['max_drawdown']:>11.2%} | {result2.metrics['max_drawdown']:>11.2%} | {result3.metrics['max_drawdown']:>11.2%}"
    )

    # 收益差异
    print("\n【收益差异分析】")
    print(f"  保守模型 vs 简单模型: {(result2.metrics['annual_return'] - result1.metrics['annual_return']):.2%}")
    print(f"  激进模型 vs 简单模型: {(result3.metrics['annual_return'] - result1.metrics['annual_return']):.2%}")
    print(f"  保守模型 vs 激进模型: {(result2.metrics['annual_return'] - result3.metrics['annual_return']):.2%}")

    # 成本差异
    avg_cost1 = np.mean([t["cost"] for t in result1.trades if t["turnover"] > 0])
    avg_cost2 = np.mean([t["cost"] for t in result2.trades if t["turnover"] > 0])
    avg_cost3 = np.mean([t["cost"] for t in result3.trades if t["turnover"] > 0])

    print("\n【平均成本对比】")
    print(f"  简单模型: {avg_cost1:.4%}")
    print(f"  保守模型: {avg_cost2:.4%} (差异: {(avg_cost2-avg_cost1):.4%})")
    print(f"  激进模型: {avg_cost3:.4%} (差异: {(avg_cost3-avg_cost1):.4%})")


if __name__ == "__main__":
    np.random.seed(42)  # 固定随机种子，保证可复现
    run_comparison()

    print("\n" + "=" * 80)
    print("✅ 对比完成")
    print("=" * 80)
