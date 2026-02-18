#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
真实场景的成本模型测试

模拟更真实的交易场景：
- 每月调仓一次（换手率30-50%）
- 每周调仓一次（换手率10-20%）
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from sage_core.backtest.simple_engine import SimpleBacktestEngine
from sage_core.backtest.types import BacktestConfig
from sage_core.backtest.cost_model import create_default_cost_model


def generate_monthly_signals(n_months: int = 12, n_stocks: int = 50) -> pd.DataFrame:
    """生成月度调仓信号"""
    dates = pd.date_range('2023-01-01', periods=n_months*21, freq='D')  # 每月21个交易日
    dates = [d.strftime('%Y%m%d') for d in dates]

    signals = []
    current_stocks = []

    for i, date in enumerate(dates):
        # 每月第一个交易日调仓
        if i % 21 == 0:
            # 保留70%的股票，替换30%
            n_keep = 7
            n_new = 3

            if current_stocks and len(current_stocks) >= n_keep:
                keep_stocks = np.random.choice(current_stocks, n_keep, replace=False).tolist()
            else:
                keep_stocks = []

            available = [f"{j:06d}.SZ" for j in range(1, n_stocks+1) if f"{j:06d}.SZ" not in keep_stocks]
            n_select = n_keep + n_new - len(keep_stocks)
            new_stocks = np.random.choice(available, n_select, replace=False).tolist()

            current_stocks = keep_stocks + new_stocks

            for stock in current_stocks:
                signals.append({
                    'trade_date': date,
                    'ts_code': stock,
                    'score': np.random.random(),
                })

    return pd.DataFrame(signals)


def generate_weekly_signals(n_weeks: int = 52, n_stocks: int = 50) -> pd.DataFrame:
    """生成周度调仓信号"""
    dates = pd.date_range('2023-01-01', periods=n_weeks*5, freq='D')  # 每周5个交易日
    dates = [d.strftime('%Y%m%d') for d in dates]

    signals = []
    current_stocks = []

    for i, date in enumerate(dates):
        # 每周第一个交易日调仓
        if i % 5 == 0:
            # 保留80%的股票，替换20%
            n_keep = 8
            n_new = 2

            if current_stocks and len(current_stocks) >= n_keep:
                keep_stocks = np.random.choice(current_stocks, n_keep, replace=False).tolist()
            else:
                keep_stocks = []

            available = [f"{j:06d}.SZ" for j in range(1, n_stocks+1) if f"{j:06d}.SZ" not in keep_stocks]
            n_select = n_keep + n_new - len(keep_stocks)
            new_stocks = np.random.choice(available, n_select, replace=False).tolist()

            current_stocks = keep_stocks + new_stocks

            for stock in current_stocks:
                signals.append({
                    'trade_date': date,
                    'ts_code': stock,
                    'score': np.random.random(),
                })

    return pd.DataFrame(signals)


def generate_returns(signals: pd.DataFrame) -> pd.DataFrame:
    """生成收益率数据"""
    dates = signals['trade_date'].unique()
    stocks = signals['ts_code'].unique()

    returns = []
    for date in dates:
        for stock in stocks:
            ret = np.random.normal(0.0005, 0.02)
            returns.append({
                'trade_date': date,
                'ts_code': stock,
                'ret': ret,
                'volatility': abs(np.random.normal(0.02, 0.005)),
                'amount': np.random.uniform(5_000_000, 50_000_000),
                'close': np.random.uniform(5, 50),
            })

    return pd.DataFrame(returns)


def run_realistic_test():
    """运行真实场景测试"""
    print("=" * 80)
    print("真实场景成本模型测试")
    print("=" * 80)

    config = BacktestConfig(
        initial_capital=10_000_000,
        cost_rate=0.0005,
        max_positions=10,
        t_plus_one=True,
        data_delay_days=2,
    )

    # 场景1：月度调仓
    print("\n" + "=" * 80)
    print("【场景1】月度调仓（每月调仓一次，替换30%持仓）")
    print("=" * 80)

    signals_monthly = generate_monthly_signals(n_months=12, n_stocks=50)
    returns_monthly = generate_returns(signals_monthly)

    print(f"  信号数据: {len(signals_monthly)} 条")
    print(f"  调仓次数: {signals_monthly['trade_date'].nunique()} 次")

    # 简单成本模型
    engine1 = SimpleBacktestEngine(config=config, use_advanced_cost=False)
    result1 = engine1.run(signals_monthly, returns_monthly)

    # 完整成本模型
    engine2 = SimpleBacktestEngine(
        config=config,
        cost_model=create_default_cost_model(),
        use_advanced_cost=True,
    )
    result2 = engine2.run(signals_monthly, returns_monthly)

    print(f"\n  简单成本模型:")
    print(f"    总收益: {result1.metrics['total_return']:>8.2%}")
    print(f"    年化收益: {result1.metrics['annual_return']:>8.2%}")
    print(f"    夏普比率: {result1.metrics['sharpe']:>8.2f}")

    avg_turnover1 = np.mean([t['turnover'] for t in result1.trades if t['turnover'] > 0])
    avg_cost1 = np.mean([t['cost'] for t in result1.trades if t['turnover'] > 0])
    print(f"    平均换手: {avg_turnover1:>8.2%}")
    print(f"    平均成本: {avg_cost1:>8.4%}")

    print(f"\n  完整成本模型:")
    print(f"    总收益: {result2.metrics['total_return']:>8.2%}")
    print(f"    年化收益: {result2.metrics['annual_return']:>8.2%}")
    print(f"    夏普比率: {result2.metrics['sharpe']:>8.2f}")

    avg_turnover2 = np.mean([t['turnover'] for t in result2.trades if t['turnover'] > 0])
    avg_cost2 = np.mean([t['cost'] for t in result2.trades if t['turnover'] > 0])
    print(f"    平均换手: {avg_turnover2:>8.2%}")
    print(f"    平均成本: {avg_cost2:>8.4%}")

    print(f"\n  成本差异: {(avg_cost2 - avg_cost1):>8.4%}")
    print(f"  收益差异: {(result2.metrics['annual_return'] - result1.metrics['annual_return']):>8.2%}")

    # 场景2：周度调仓
    print("\n" + "=" * 80)
    print("【场景2】周度调仓（每周调仓一次，替换20%持仓）")
    print("=" * 80)

    signals_weekly = generate_weekly_signals(n_weeks=52, n_stocks=50)
    returns_weekly = generate_returns(signals_weekly)

    print(f"  信号数据: {len(signals_weekly)} 条")
    print(f"  调仓次数: {signals_weekly['trade_date'].nunique()} 次")

    # 简单成本模型
    engine3 = SimpleBacktestEngine(config=config, use_advanced_cost=False)
    result3 = engine3.run(signals_weekly, returns_weekly)

    # 完整成本模型
    engine4 = SimpleBacktestEngine(
        config=config,
        cost_model=create_default_cost_model(),
        use_advanced_cost=True,
    )
    result4 = engine4.run(signals_weekly, returns_weekly)

    print(f"\n  简单成本模型:")
    print(f"    总收益: {result3.metrics['total_return']:>8.2%}")
    print(f"    年化收益: {result3.metrics['annual_return']:>8.2%}")
    print(f"    夏普比率: {result3.metrics['sharpe']:>8.2f}")

    avg_turnover3 = np.mean([t['turnover'] for t in result3.trades if t['turnover'] > 0])
    avg_cost3 = np.mean([t['cost'] for t in result3.trades if t['turnover'] > 0])
    print(f"    平均换手: {avg_turnover3:>8.2%}")
    print(f"    平均成本: {avg_cost3:>8.4%}")

    print(f"\n  完整成本模型:")
    print(f"    总收益: {result4.metrics['total_return']:>8.2%}")
    print(f"    年化收益: {result4.metrics['annual_return']:>8.2%}")
    print(f"    夏普比率: {result4.metrics['sharpe']:>8.2f}")

    avg_turnover4 = np.mean([t['turnover'] for t in result4.trades if t['turnover'] > 0])
    avg_cost4 = np.mean([t['cost'] for t in result4.trades if t['turnover'] > 0])
    print(f"    平均换手: {avg_turnover4:>8.2%}")
    print(f"    平均成本: {avg_cost4:>8.4%}")

    print(f"\n  成本差异: {(avg_cost4 - avg_cost3):>8.4%}")
    print(f"  收益差异: {(result4.metrics['annual_return'] - result3.metrics['annual_return']):>8.2%}")


if __name__ == "__main__":
    np.random.seed(42)
    run_realistic_test()

    print("\n" + "=" * 80)
    print("✅ 测试完成")
    print("=" * 80)
