#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试交易成本模型
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sage_core.backtest.cost_model import (
    TradingCostModel,
    CostModelConfig,
    create_default_cost_model,
    create_aggressive_cost_model,
)


def test_single_trade_cost():
    """测试单笔交易成本"""
    print("=" * 80)
    print("测试1：单笔交易成本计算")
    print("=" * 80)

    model = create_default_cost_model()

    # 场景1：买入100万，低波动率，高流动性
    print("\n场景1：买入100万（低波动率2%，日成交1000万）")
    cost = model.calculate_total_cost(
        trade_value=1_000_000,
        is_buy=True,
        volatility=0.02,
        daily_volume=10_000_000,
        avg_price=10.0,
    )
    print(f"  固定成本: {cost['fixed_cost']:.2f} 元 ({cost['fixed_cost']/1_000_000*100:.4f}%)")
    print(f"  滑点成本: {cost['slippage_cost']:.2f} 元 ({cost['slippage_cost']/1_000_000*100:.4f}%)")
    print(f"  市场冲击: {cost['market_impact_cost']:.2f} 元 ({cost['market_impact_cost']/1_000_000*100:.4f}%)")
    print(f"  总成本: {cost['total_cost']:.2f} 元 ({cost['total_cost_rate']*100:.4f}%)")

    # 场景2：卖出100万，高波动率，低流动性
    print("\n场景2：卖出100万（高波动率5%，日成交200万）")
    cost = model.calculate_total_cost(
        trade_value=1_000_000,
        is_buy=False,
        volatility=0.05,
        daily_volume=2_000_000,
        avg_price=10.0,
    )
    print(f"  固定成本: {cost['fixed_cost']:.2f} 元 ({cost['fixed_cost']/1_000_000*100:.4f}%)")
    print(f"  滑点成本: {cost['slippage_cost']:.2f} 元 ({cost['slippage_cost']/1_000_000*100:.4f}%)")
    print(f"  市场冲击: {cost['market_impact_cost']:.2f} 元 ({cost['market_impact_cost']/1_000_000*100:.4f}%)")
    print(f"  总成本: {cost['total_cost']:.2f} 元 ({cost['total_cost_rate']*100:.4f}%)")


def test_turnover_cost():
    """测试换手成本"""
    print("\n" + "=" * 80)
    print("测试2：换手成本计算")
    print("=" * 80)

    model = create_default_cost_model()
    portfolio_value = 10_000_000  # 1000万组合

    turnovers = [0.1, 0.3, 0.5, 1.0, 2.0]  # 不同换手率

    print(f"\n组合价值: {portfolio_value:,.0f} 元")
    print("\n换手率 | 交易金额 | 成本率 | 成本金额")
    print("-" * 60)

    for turnover in turnovers:
        cost_rate = model.calculate_turnover_cost(
            turnover=turnover,
            portfolio_value=portfolio_value,
        )
        trade_value = turnover * portfolio_value
        cost_value = cost_rate * portfolio_value

        print(f"{turnover:>6.1%} | {trade_value:>10,.0f} | {cost_rate:>6.4%} | {cost_value:>10,.2f}")


def test_cost_model_comparison():
    """对比不同成本模型"""
    print("\n" + "=" * 80)
    print("测试3：成本模型对比（保守 vs 激进）")
    print("=" * 80)

    default_model = create_default_cost_model()
    aggressive_model = create_aggressive_cost_model()

    trade_value = 1_000_000
    volatility = 0.03
    daily_volume = 5_000_000

    print(f"\n交易金额: {trade_value:,.0f} 元")
    print(f"波动率: {volatility:.2%}")
    print(f"日成交额: {daily_volume:,.0f} 元")

    print("\n【买入成本对比】")
    print("-" * 60)

    default_cost = default_model.calculate_total_cost(
        trade_value, is_buy=True, volatility=volatility, daily_volume=daily_volume
    )
    aggressive_cost = aggressive_model.calculate_total_cost(
        trade_value, is_buy=True, volatility=volatility, daily_volume=daily_volume
    )

    print(f"{'成本项':<15} | {'保守模型':>12} | {'激进模型':>12} | {'差异':>10}")
    print("-" * 60)
    print(f"{'固定成本':<15} | {default_cost['fixed_cost']:>10,.2f} | {aggressive_cost['fixed_cost']:>10,.2f} | {(default_cost['fixed_cost']-aggressive_cost['fixed_cost']):>8,.2f}")
    print(f"{'滑点成本':<15} | {default_cost['slippage_cost']:>10,.2f} | {aggressive_cost['slippage_cost']:>10,.2f} | {(default_cost['slippage_cost']-aggressive_cost['slippage_cost']):>8,.2f}")
    print(f"{'市场冲击':<15} | {default_cost['market_impact_cost']:>10,.2f} | {aggressive_cost['market_impact_cost']:>10,.2f} | {(default_cost['market_impact_cost']-aggressive_cost['market_impact_cost']):>8,.2f}")
    print(f"{'总成本':<15} | {default_cost['total_cost']:>10,.2f} | {aggressive_cost['total_cost']:>10,.2f} | {(default_cost['total_cost']-aggressive_cost['total_cost']):>8,.2f}")
    print(f"{'成本率':<15} | {default_cost['total_cost_rate']:>11.4%} | {aggressive_cost['total_cost_rate']:>11.4%} | {(default_cost['total_cost_rate']-aggressive_cost['total_cost_rate']):>9.4%}")


def test_liquidity_constraint():
    """测试流动性约束"""
    print("\n" + "=" * 80)
    print("测试4：流动性约束检查")
    print("=" * 80)

    model = create_default_cost_model()

    test_cases = [
        (1_000_000, 10_000_000, "正常流动性"),
        (1_000_000, 2_000_000, "低流动性（参与率50%）"),
        (500_000, 500_000, "极低流动性（参与率100%）"),
        (100_000, 10_000_000, "高流动性（参与率1%）"),
    ]

    print("\n交易金额 | 日成交额 | 参与率 | 是否通过 | 说明")
    print("-" * 80)

    for trade_value, daily_volume, desc in test_cases:
        passed = model.check_liquidity_constraint(trade_value, daily_volume)
        participation = trade_value / daily_volume if daily_volume > 0 else 1.0
        status = "✓ 通过" if passed else "✗ 不通过"

        print(f"{trade_value:>10,.0f} | {daily_volume:>10,.0f} | {participation:>6.2%} | {status:>8} | {desc}")


def test_cost_sensitivity():
    """测试成本敏感性分析"""
    print("\n" + "=" * 80)
    print("测试5：成本敏感性分析")
    print("=" * 80)

    model = create_default_cost_model()
    base_trade_value = 1_000_000

    # 波动率敏感性
    print("\n【波动率敏感性】（日成交500万）")
    print("波动率 | 总成本率 | 滑点成本率 | 市场冲击率")
    print("-" * 60)

    for vol in [0.01, 0.02, 0.03, 0.05, 0.10]:
        cost = model.calculate_total_cost(
            base_trade_value, is_buy=True, volatility=vol, daily_volume=5_000_000
        )
        print(f"{vol:>6.2%} | {cost['total_cost_rate']:>10.4%} | {cost['slippage_cost']/base_trade_value:>12.4%} | {cost['market_impact_cost']/base_trade_value:>12.4%}")

    # 流动性敏感性
    print("\n【流动性敏感性】（波动率3%）")
    print("日成交额 | 参与率 | 总成本率 | 市场冲击率")
    print("-" * 60)

    for daily_vol in [1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000]:
        cost = model.calculate_total_cost(
            base_trade_value, is_buy=True, volatility=0.03, daily_volume=daily_vol
        )
        participation = base_trade_value / daily_vol
        print(f"{daily_vol:>10,.0f} | {participation:>6.2%} | {cost['total_cost_rate']:>10.4%} | {cost['market_impact_cost']/base_trade_value:>12.4%}")


if __name__ == "__main__":
    test_single_trade_cost()
    test_turnover_cost()
    test_cost_model_comparison()
    test_liquidity_constraint()
    test_cost_sensitivity()

    print("\n" + "=" * 80)
    print("✅ 所有测试完成")
    print("=" * 80)
