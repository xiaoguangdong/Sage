#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分析成本模型的成本构成
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sage_core.backtest.cost_model import create_default_cost_model


def analyze_cost_breakdown():
    """分析成本构成"""
    print("=" * 80)
    print("成本模型构成分析")
    print("=" * 80)

    model = create_default_cost_model()

    # 测试场景：100万交易，不同波动率和流动性
    test_cases = [
        {
            "name": "高流动性低波动",
            "trade_value": 1_000_000,
            "volatility": 0.02,
            "daily_volume": 20_000_000,
            "avg_price": 10.0,
        },
        {
            "name": "中等流动性中等波动",
            "trade_value": 1_000_000,
            "volatility": 0.03,
            "daily_volume": 10_000_000,
            "avg_price": 10.0,
        },
        {
            "name": "低流动性高波动",
            "trade_value": 1_000_000,
            "volatility": 0.05,
            "daily_volume": 5_000_000,
            "avg_price": 10.0,
        },
    ]

    for case in test_cases:
        print(f"\n【{case['name']}】")
        print(f"  交易金额: {case['trade_value']:,.0f} 元")
        print(f"  波动率: {case['volatility']:.2%}")
        print(f"  日成交额: {case['daily_volume']:,.0f} 元")
        print(f"  参与率: {case['trade_value']/case['daily_volume']:.2%}")

        # 买入成本
        buy_cost = model.calculate_total_cost(
            case['trade_value'],
            is_buy=True,
            volatility=case['volatility'],
            daily_volume=case['daily_volume'],
            avg_price=case['avg_price'],
        )

        print(f"\n  买入成本:")
        print(f"    固定成本: {buy_cost['fixed_cost']:>10,.2f} 元 ({buy_cost['fixed_cost']/case['trade_value']*100:>6.4f}%)")
        print(f"    滑点成本: {buy_cost['slippage_cost']:>10,.2f} 元 ({buy_cost['slippage_cost']/case['trade_value']*100:>6.4f}%)")
        print(f"    市场冲击: {buy_cost['market_impact_cost']:>10,.2f} 元 ({buy_cost['market_impact_cost']/case['trade_value']*100:>6.4f}%)")
        print(f"    总成本:   {buy_cost['total_cost']:>10,.2f} 元 ({buy_cost['total_cost_rate']*100:>6.4f}%)")

        # 卖出成本
        sell_cost = model.calculate_total_cost(
            case['trade_value'],
            is_buy=False,
            volatility=case['volatility'],
            daily_volume=case['daily_volume'],
            avg_price=case['avg_price'],
        )

        print(f"\n  卖出成本:")
        print(f"    固定成本: {sell_cost['fixed_cost']:>10,.2f} 元 ({sell_cost['fixed_cost']/case['trade_value']*100:>6.4f}%)")
        print(f"    滑点成本: {sell_cost['slippage_cost']:>10,.2f} 元 ({sell_cost['slippage_cost']/case['trade_value']*100:>6.4f}%)")
        print(f"    市场冲击: {sell_cost['market_impact_cost']:>10,.2f} 元 ({sell_cost['market_impact_cost']/case['trade_value']*100:>6.4f}%)")
        print(f"    总成本:   {sell_cost['total_cost']:>10,.2f} 元 ({sell_cost['total_cost_rate']*100:>6.4f}%)")

        # 双边成本
        total_cost = buy_cost['total_cost'] + sell_cost['total_cost']
        print(f"\n  双边总成本: {total_cost:>10,.2f} 元 ({total_cost/case['trade_value']*100:>6.4f}%)")

    # 分析换手率对成本的影响
    print("\n" + "=" * 80)
    print("换手率对成本的影响")
    print("=" * 80)

    portfolio_value = 10_000_000
    turnovers = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0]

    print(f"\n组合价值: {portfolio_value:,.0f} 元")
    print(f"\n换手率 | 交易金额 | 成本率 | 成本金额 | 年化成本(252天)")
    print("-" * 80)

    for turnover in turnovers:
        cost_rate = model.calculate_turnover_cost(
            turnover=turnover,
            portfolio_value=portfolio_value,
        )
        trade_value = turnover * portfolio_value
        cost_value = cost_rate * portfolio_value
        annual_cost = cost_rate * 252  # 假设每天都换手

        print(f"{turnover:>6.1f} | {trade_value:>10,.0f} | {cost_rate:>6.4%} | {cost_value:>10,.2f} | {annual_cost:>14.2%}")


if __name__ == "__main__":
    analyze_cost_breakdown()

    print("\n" + "=" * 80)
    print("✅ 分析完成")
    print("=" * 80)
