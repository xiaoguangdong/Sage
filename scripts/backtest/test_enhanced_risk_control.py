#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试增强风险控制模块
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from sage_core.portfolio.enhanced_risk_control import (
    EnhancedRiskControl,
    RiskControlConfig,
)


def test_dynamic_position():
    """测试动态仓位计算"""
    print("=" * 80)
    print("测试1：基于confidence的动态仓位")
    print("=" * 80)

    config = RiskControlConfig(
        base_position=0.6,
        max_position=1.0,
        min_position=0.3,
    )
    risk_control = EnhancedRiskControl(config)

    # 测试不同confidence下的仓位
    confidences = [0.2, 0.4, 0.6, 0.8, 1.0]

    print("\n【基础场景】无回撤，无冲击")
    print(f"{'Confidence':<12} | {'目标仓位':<10}")
    print("-" * 30)

    for conf in confidences:
        position = risk_control.compute_dynamic_position(conf)
        print(f"{conf:<12.1f} | {position:<10.2%}")

    # 测试回撤影响
    print("\n【回撤场景】不同回撤水平")
    print(f"{'Confidence':<12} | {'回撤':<10} | {'目标仓位':<10}")
    print("-" * 40)

    drawdowns = [-0.05, -0.10, -0.12, -0.15]
    for dd in drawdowns:
        position = risk_control.compute_dynamic_position(
            confidence=0.8,
            current_drawdown=dd,
        )
        print(f"{0.8:<12.1f} | {dd:<10.2%} | {position:<10.2%}")

    # 测试单日冲击
    print("\n【冲击场景】单日大跌")
    print(f"{'Confidence':<12} | {'单日收益':<10} | {'目标仓位':<10}")
    print("-" * 40)

    daily_returns = [-0.01, -0.02, -0.03, -0.05]
    for ret in daily_returns:
        position = risk_control.compute_dynamic_position(
            confidence=0.8,
            daily_return=ret,
        )
        print(f"{0.8:<12.1f} | {ret:<10.2%} | {position:<10.2%}")


def test_atr_stop_loss():
    """测试ATR止损"""
    print("\n" + "=" * 80)
    print("测试2：ATR动态止损")
    print("=" * 80)

    config = RiskControlConfig(
        atr_stop_loss_multiplier=2.0,
        atr_period=14,
    )
    risk_control = EnhancedRiskControl(config)

    # 生成模拟价格数据
    np.random.seed(42)
    n_days = 30
    base_price = 10.0

    prices = [base_price]
    for _ in range(n_days - 1):
        change = np.random.normal(0, 0.02)
        prices.append(prices[-1] * (1 + change))

    df = pd.DataFrame({
        'close': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
    })

    entry_price = prices[0]
    stop_price = risk_control.compute_atr_stop_loss(
        df['close'],
        df['high'],
        df['low'],
        entry_price,
    )

    print(f"\n入场价格: {entry_price:.2f}")
    print(f"ATR止损价: {stop_price:.2f}")
    print(f"止损幅度: {(stop_price/entry_price - 1):.2%}")
    print(f"当前价格: {prices[-1]:.2f}")
    print(f"当前盈亏: {(prices[-1]/entry_price - 1):.2%}")

    # 测试止损触发
    print("\n【模拟价格下跌】")
    test_prices = [entry_price * (1 - i * 0.02) for i in range(10)]
    print(f"{'价格':<10} | {'相对入场':<12} | {'是否止损':<10}")
    print("-" * 40)

    for price in test_prices:
        is_stop = price <= stop_price
        print(f"{price:<10.2f} | {(price/entry_price-1):<12.2%} | {'✓ 止损' if is_stop else '持有':<10}")


def test_industry_stop_loss():
    """测试行业止损"""
    print("\n" + "=" * 80)
    print("测试3：行业层面止损")
    print("=" * 80)

    config = RiskControlConfig(
        industry_drawdown_threshold=-0.15,
    )
    risk_control = EnhancedRiskControl(config)

    # 模拟行业收益率
    industries = ['电子', '医药', '消费', '金融']

    print("\n【场景1】行业正常波动")
    industry_returns_1 = {
        '电子': 0.05,
        '医药': -0.08,
        '消费': 0.02,
        '金融': -0.10,
    }

    stop_industries_1 = risk_control.check_industry_stop_loss(industry_returns_1)
    print(f"{'行业':<10} | {'累计收益':<12} | {'状态':<10}")
    print("-" * 40)
    for ind, ret in industry_returns_1.items():
        status = '✗ 清仓' if ind in stop_industries_1 else '持有'
        print(f"{ind:<10} | {ret:<12.2%} | {status:<10}")

    print("\n【场景2】行业大幅回撤")
    industry_returns_2 = {
        '电子': -0.18,  # 触发止损
        '医药': -0.12,
        '消费': -0.16,  # 触发止损
        '金融': -0.08,
    }

    # 重置风控状态
    risk_control = EnhancedRiskControl(config)
    stop_industries_2 = risk_control.check_industry_stop_loss(industry_returns_2)

    print(f"{'行业':<10} | {'累计收益':<12} | {'状态':<10}")
    print("-" * 40)
    for ind, ret in industry_returns_2.items():
        status = '✗ 清仓' if ind in stop_industries_2 else '持有'
        print(f"{ind:<10} | {ret:<12.2%} | {status:<10}")


def test_position_limits():
    """测试仓位限制"""
    print("\n" + "=" * 80)
    print("测试4：仓位限制")
    print("=" * 80)

    config = RiskControlConfig(
        max_single_position=0.10,
        max_industry_exposure=0.30,
    )
    risk_control = EnhancedRiskControl(config)

    # 创建测试权重
    stocks = [f"stock_{i}" for i in range(10)]
    industries = pd.Series(
        ['电子', '电子', '电子', '电子', '医药', '医药', '消费', '消费', '金融', '金融'],
        index=stocks
    )

    # 原始权重（有些超限）
    original_weights = pd.Series([0.15, 0.12, 0.10, 0.08, 0.10, 0.10, 0.10, 0.10, 0.08, 0.07], index=stocks)

    print("\n【原始权重】")
    print(f"{'股票':<10} | {'行业':<10} | {'权重':<10}")
    print("-" * 40)
    for stock, weight in original_weights.items():
        ind = industries[stock]
        print(f"{stock:<10} | {ind:<10} | {weight:<10.2%}")

    print(f"\n行业暴露:")
    for ind in industries.unique():
        ind_weight = original_weights[industries == ind].sum()
        print(f"  {ind}: {ind_weight:.2%}")

    # 应用限制
    adjusted_weights = risk_control.apply_position_limits(original_weights, industries)

    print("\n【调整后权重】")
    print(f"{'股票':<10} | {'行业':<10} | {'原始':<10} | {'调整后':<10}")
    print("-" * 50)
    for stock in stocks:
        ind = industries[stock]
        orig = original_weights[stock]
        adj = adjusted_weights[stock]
        print(f"{stock:<10} | {ind:<10} | {orig:<10.2%} | {adj:<10.2%}")

    print(f"\n调整后行业暴露:")
    for ind in industries.unique():
        ind_weight = adjusted_weights[industries == ind].sum()
        print(f"  {ind}: {ind_weight:.2%}")


def test_comprehensive_scenario():
    """综合场景测试"""
    print("\n" + "=" * 80)
    print("测试5：综合场景模拟")
    print("=" * 80)

    config = RiskControlConfig(
        base_position=0.6,
        max_position=1.0,
        min_position=0.3,
    )
    risk_control = EnhancedRiskControl(config)

    # 模拟一个完整的交易周期
    scenarios = [
        {'day': 1, 'confidence': 0.8, 'drawdown': 0.0, 'daily_ret': 0.01, 'desc': '牛市初期'},
        {'day': 5, 'confidence': 0.9, 'drawdown': 0.05, 'daily_ret': 0.02, 'desc': '牛市加速'},
        {'day': 10, 'confidence': 0.7, 'drawdown': 0.03, 'daily_ret': -0.01, 'desc': '震荡调整'},
        {'day': 15, 'confidence': 0.5, 'drawdown': -0.05, 'daily_ret': -0.02, 'desc': '回调'},
        {'day': 20, 'confidence': 0.4, 'drawdown': -0.11, 'daily_ret': -0.03, 'desc': '深度回调'},
        {'day': 25, 'confidence': 0.3, 'drawdown': -0.13, 'daily_ret': -0.01, 'desc': '触发分档降仓'},
        {'day': 30, 'confidence': 0.6, 'drawdown': -0.08, 'daily_ret': 0.02, 'desc': '反弹'},
    ]

    print(f"\n{'日期':<6} | {'Conf':<6} | {'回撤':<8} | {'日收益':<8} | {'目标仓位':<10} | {'说明':<15}")
    print("-" * 80)

    for scenario in scenarios:
        position = risk_control.compute_dynamic_position(
            confidence=scenario['confidence'],
            current_drawdown=scenario['drawdown'],
            daily_return=scenario['daily_ret'],
        )
        print(
            f"{scenario['day']:<6} | {scenario['confidence']:<6.1f} | "
            f"{scenario['drawdown']:<8.2%} | {scenario['daily_ret']:<8.2%} | "
            f"{position:<10.2%} | {scenario['desc']:<15}"
        )


if __name__ == "__main__":
    test_dynamic_position()
    test_atr_stop_loss()
    test_industry_stop_loss()
    test_position_limits()
    test_comprehensive_scenario()

    print("\n" + "=" * 80)
    print("✅ 所有测试完成")
    print("=" * 80)
