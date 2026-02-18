#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试混合选股器 + 组合管理器
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from sage_core.stock_selection.hybrid_stock_selector import HybridStockSelector
from sage_core.portfolio.portfolio_manager import PortfolioManager


def generate_mock_data(n_stocks: int = 1000) -> pd.DataFrame:
    """生成模拟数据用于测试

    Args:
        n_stocks: 股票数量

    Returns:
        模拟的股票特征数据
    """
    np.random.seed(42)

    data = {
        'ts_code': [f'{i:06d}.SZ' for i in range(n_stocks)],
        'trade_date': ['20241231'] * n_stocks,
        'industry': np.random.choice(['电子', '医药', '计算机', '机械', '化工'], n_stocks),

        # 价值股特征
        'roe': np.random.normal(0.12, 0.08, n_stocks).clip(0, 0.50),
        'roe_5y_avg': np.random.normal(0.10, 0.06, n_stocks).clip(0, 0.40),
        'debt_ratio': np.random.normal(0.50, 0.20, n_stocks).clip(0, 1.0),
        'interest_coverage': np.random.lognormal(1.5, 1.0, n_stocks).clip(0, 100),
        'consecutive_dividend': np.random.randint(0, 15, n_stocks),
        'dividend_yield': np.random.normal(0.02, 0.015, n_stocks).clip(0, 0.10),
        'pe_relative': np.random.normal(1.0, 0.3, n_stocks).clip(0.3, 3.0),
        'fund_holders': np.random.randint(0, 100, n_stocks),
        'inst_holding_change': np.random.normal(0.02, 0.10, n_stocks).clip(-0.50, 0.50),
        'revenue_growth': np.random.normal(0.10, 0.20, n_stocks).clip(-0.50, 1.0),

        # 成长股特征
        'revenue_cagr_3y': np.random.normal(0.20, 0.15, n_stocks).clip(-0.50, 1.0),
        'profit_cagr_3y': np.random.normal(0.15, 0.20, n_stocks).clip(-0.50, 1.0),
        'rd_ratio': np.random.normal(0.06, 0.04, n_stocks).clip(0, 0.20),
        'gross_margin': np.random.normal(0.30, 0.15, n_stocks).clip(0, 0.80),
        'gross_margin_trend': np.random.normal(0.01, 0.03, n_stocks).clip(-0.10, 0.10),
        'asset_turnover': np.random.normal(1.0, 0.5, n_stocks).clip(0.1, 5.0),
        'industry_rank': np.random.randint(1, 50, n_stocks),

        # 辅助特征
        'is_st': np.random.choice([True, False], n_stocks, p=[0.05, 0.95]),
        'avg_amount': np.random.lognormal(18, 1.5, n_stocks),  # 日均成交额

        # 目标变量（未来6个月收益率）
        'return_6m': np.random.normal(0.05, 0.30, n_stocks),
    }

    df = pd.DataFrame(data)

    # 添加一些相关性（让特征更真实）
    # ROE高的股票，未来收益率略高
    df.loc[df['roe'] > 0.15, 'return_6m'] += 0.05
    # 高增长的股票，未来收益率更高
    df.loc[df['revenue_cagr_3y'] > 0.25, 'return_6m'] += 0.08
    # ST股票收益率更低
    df.loc[df['is_st'] == True, 'return_6m'] -= 0.15

    return df


def test_integrated_system():
    """测试完整系统：选股器 + 组合管理器"""
    print("=" * 60)
    print("测试完整系统：选股器 + 组合管理器")
    print("=" * 60)

    # 1. 生成模拟数据
    df_train = generate_mock_data(n_stocks=2000)
    df_test = generate_mock_data(n_stocks=500)

    print(f"\n训练集: {len(df_train)} 只股票")
    print(f"测试集: {len(df_test)} 只股票")

    # 2. 创建价值股选股器
    print("\n" + "=" * 60)
    print("【步骤1】训练价值股选股器")
    print("=" * 60)
    value_selector = HybridStockSelector(
        selector_type='value',
        data_root=Path('/Users/dongxg/SourceCode/Sage/data/tushare'),
        model_path=Path('/Users/dongxg/SourceCode/Sage/models/value_selector.pkl'),
    )
    value_metrics = value_selector.train(df_train, target_col='return_6m')
    print(f"R²: {value_metrics['r2']:.4f}, 样本数: {value_metrics['n_samples']}")

    # 3. 创建成长股选股器
    print("\n" + "=" * 60)
    print("【步骤2】训练成长股选股器")
    print("=" * 60)
    growth_selector = HybridStockSelector(
        selector_type='growth',
        data_root=Path('/Users/dongxg/SourceCode/Sage/data/tushare'),
        model_path=Path('/Users/dongxg/SourceCode/Sage/models/growth_selector.pkl'),
    )
    growth_metrics = growth_selector.train(df_train, target_col='return_6m')
    print(f"R²: {growth_metrics['r2']:.4f}, 样本数: {growth_metrics['n_samples']}")

    # 4. 在测试集上选股
    print("\n" + "=" * 60)
    print("【步骤3】在测试集上选股")
    print("=" * 60)
    value_candidates = value_selector.select(df_test, top_n=50)
    growth_candidates = growth_selector.select(df_test, top_n=50)

    # 5. 创建组合管理器
    print("\n" + "=" * 60)
    print("【步骤4】构建投资组合")
    print("=" * 60)
    portfolio_manager = PortfolioManager(
        long_term_ratio=0.70,
        short_term_ratio=0.30,
        value_growth_ratio=0.50,
        max_industry_ratio=0.30,
    )

    # 6. 构建组合（暂时没有动量股）
    portfolio = portfolio_manager.construct_portfolio(
        value_candidates=value_candidates,
        growth_candidates=growth_candidates,
        momentum_candidates=None,  # 暂时没有短线策略
        total_positions=10,
    )

    # 7. 展示组合
    if not portfolio.empty:
        print("\n" + "=" * 60)
        print("【最终组合】")
        print("=" * 60)
        display_cols = ['ts_code', 'industry', 'strategy_type', 'score', 'weight']
        print(portfolio[display_cols].to_string(index=False))

        # 8. 评估组合效果
        print("\n" + "=" * 60)
        print("【组合效果评估】")
        print("=" * 60)
        portfolio_return = (portfolio['return_6m'] * portfolio['weight']).sum()
        benchmark_return = df_test['return_6m'].mean()

        print(f"组合预期收益: {portfolio_return:.2%}")
        print(f"基准平均收益: {benchmark_return:.2%}")
        print(f"超额收益: {(portfolio_return - benchmark_return):.2%}")

        # 按策略类型统计
        print("\n按策略类型统计:")
        for strategy in portfolio['strategy_type'].unique():
            strategy_stocks = portfolio[portfolio['strategy_type'] == strategy]
            strategy_return = (strategy_stocks['return_6m'] * strategy_stocks['weight']).sum()
            strategy_weight = strategy_stocks['weight'].sum()
            print(f"  {strategy}: 权重{strategy_weight:.1%}, 预期收益{strategy_return:.2%}")

    else:
        print("警告: 没有构建出组合")

    # 9. 测试调仓逻辑
    print("\n" + "=" * 60)
    print("【测试调仓逻辑】")
    print("=" * 60)

    # 模拟一个月后重新选股
    df_rebalance = generate_mock_data(n_stocks=500)
    value_candidates_new = value_selector.select(df_rebalance, top_n=50)
    growth_candidates_new = growth_selector.select(df_rebalance, top_n=50)

    # 全部调仓
    print("\n场景1: 季度调仓（全部重新构建）")
    new_portfolio = portfolio_manager.rebalance(
        current_portfolio=portfolio,
        new_candidates={
            'value': value_candidates_new,
            'growth': growth_candidates_new,
        },
        rebalance_type='full',
    )
    print(f"调仓后持仓数: {len(new_portfolio)}")

    # 10. 测试风险控制
    print("\n" + "=" * 60)
    print("【测试风险控制】")
    print("=" * 60)

    print("\n场景1: 回撤-10%（正常持仓）")
    controlled_portfolio = portfolio_manager.apply_risk_control(portfolio, current_drawdown=-0.10)
    print(f"总权重: {controlled_portfolio['weight'].sum():.1%}")

    print("\n场景2: 回撤-18%（减仓50%）")
    controlled_portfolio = portfolio_manager.apply_risk_control(portfolio, current_drawdown=-0.18)
    print(f"总权重: {controlled_portfolio['weight'].sum():.1%}")

    print("\n场景3: 回撤-30%（清仓）")
    controlled_portfolio = portfolio_manager.apply_risk_control(portfolio, current_drawdown=-0.30)
    print(f"总权重: {controlled_portfolio['weight'].sum():.1%}")


if __name__ == "__main__":
    test_integrated_system()

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
