#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
组合管理器

职责：
1. 整合多个选股器的输出（长线价值股、长线成长股、短线动量股）
2. 资金分配（70%长线 + 30%短线）
3. 组合构建约束（行业分散、流动性、风险控制）
4. 调仓逻辑（长线季度调仓、短线周度调仓）
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime


class PortfolioManager:
    """组合管理器

    架构：
    - 长线策略（70%资金）：价值股 + 成长股
    - 短线策略（30%资金）：动量股
    - 统一风险控制：回撤-15%减仓50%，-25%清仓
    """

    def __init__(
        self,
        long_term_ratio: float = 0.70,
        short_term_ratio: float = 0.30,
        value_growth_ratio: float = 0.50,  # 长线中价值/成长的配比
        max_industry_ratio: float = 0.30,
        min_avg_amount: float = 1e8,
    ):
        """初始化组合管理器

        Args:
            long_term_ratio: 长线资金占比（默认70%）
            short_term_ratio: 短线资金占比（默认30%）
            value_growth_ratio: 长线中价值股占比（默认50%）
            max_industry_ratio: 单行业最大占比（默认30%）
            min_avg_amount: 最小日均成交额（默认1亿）
        """
        self.long_term_ratio = long_term_ratio
        self.short_term_ratio = short_term_ratio
        self.value_growth_ratio = value_growth_ratio
        self.max_industry_ratio = max_industry_ratio
        self.min_avg_amount = min_avg_amount

        # 验证配比
        assert abs(long_term_ratio + short_term_ratio - 1.0) < 1e-6, \
            "长短线资金占比之和必须为1"

    def construct_portfolio(
        self,
        value_candidates: pd.DataFrame,
        growth_candidates: pd.DataFrame,
        momentum_candidates: Optional[pd.DataFrame] = None,
        total_positions: int = 10,
    ) -> pd.DataFrame:
        """构建投资组合

        Args:
            value_candidates: 价值股候选池（带score列）
            growth_candidates: 成长股候选池（带score列）
            momentum_candidates: 动量股候选池（可选，带score列）
            total_positions: 总持仓数量

        Returns:
            最终投资组合（包含股票代码、权重、策略类型）
        """
        # 1. 计算各策略持仓数量
        long_term_positions = int(total_positions * self.long_term_ratio)
        short_term_positions = total_positions - long_term_positions

        value_positions = int(long_term_positions * self.value_growth_ratio)
        growth_positions = long_term_positions - value_positions

        print(f"\n组合配置:")
        print(f"  总持仓: {total_positions} 只")
        print(f"  长线: {long_term_positions} 只 (价值{value_positions} + 成长{growth_positions})")
        print(f"  短线: {short_term_positions} 只")

        # 2. 从各候选池中选股
        portfolio_parts = []

        # 2.1 价值股
        if not value_candidates.empty:
            value_stocks = self._select_from_pool(
                value_candidates,
                n_stocks=value_positions,
                strategy_type='value',
            )
            portfolio_parts.append(value_stocks)
            print(f"  价值股选出: {len(value_stocks)} 只")

        # 2.2 成长股
        if not growth_candidates.empty:
            growth_stocks = self._select_from_pool(
                growth_candidates,
                n_stocks=growth_positions,
                strategy_type='growth',
            )
            portfolio_parts.append(growth_stocks)
            print(f"  成长股选出: {len(growth_stocks)} 只")

        # 2.3 动量股（短线）
        if momentum_candidates is not None and not momentum_candidates.empty:
            momentum_stocks = self._select_from_pool(
                momentum_candidates,
                n_stocks=short_term_positions,
                strategy_type='momentum',
            )
            portfolio_parts.append(momentum_stocks)
            print(f"  动量股选出: {len(momentum_stocks)} 只")

        # 3. 合并组合
        if not portfolio_parts:
            print("警告: 没有选出任何股票")
            return pd.DataFrame()

        portfolio = pd.concat(portfolio_parts, ignore_index=True)

        # 4. 计算权重
        portfolio = self._calculate_weights(portfolio)

        # 5. 行业分散检查
        self._check_industry_concentration(portfolio)

        return portfolio

    def _select_from_pool(
        self,
        candidates: pd.DataFrame,
        n_stocks: int,
        strategy_type: str,
    ) -> pd.DataFrame:
        """从候选池中选股

        Args:
            candidates: 候选股票池（带score列）
            n_stocks: 目标持仓数量
            strategy_type: 策略类型（'value', 'growth', 'momentum'）

        Returns:
            选出的股票
        """
        selected = []
        industry_count = {}
        max_per_industry = max(1, int(n_stocks * self.max_industry_ratio))

        for _, stock in candidates.iterrows():
            # 流动性约束
            if 'avg_amount' in stock and stock['avg_amount'] < self.min_avg_amount:
                continue

            # 行业分散约束
            industry = stock.get('industry', 'Unknown')
            if industry_count.get(industry, 0) >= max_per_industry:
                continue

            # 添加策略类型标签
            stock_dict = stock.to_dict()
            stock_dict['strategy_type'] = strategy_type
            selected.append(stock_dict)

            industry_count[industry] = industry_count.get(industry, 0) + 1

            if len(selected) >= n_stocks:
                break

        return pd.DataFrame(selected)

    def _calculate_weights(self, portfolio: pd.DataFrame) -> pd.DataFrame:
        """计算组合权重

        根据策略类型分配权重：
        - 长线策略（价值+成长）：70%资金
        - 短线策略（动量）：30%资金
        - 同一策略内等权重

        Args:
            portfolio: 组合DataFrame（包含strategy_type列）

        Returns:
            带有weight列的组合
        """
        portfolio = portfolio.copy()

        # 统计各策略股票数量
        strategy_counts = portfolio['strategy_type'].value_counts()

        # 计算每只股票的权重
        weights = []
        for _, stock in portfolio.iterrows():
            strategy = stock['strategy_type']

            if strategy in ['value', 'growth']:
                # 长线策略：70%资金 / 长线股票数量
                long_term_stocks = strategy_counts.get('value', 0) + strategy_counts.get('growth', 0)
                weight = self.long_term_ratio / long_term_stocks if long_term_stocks > 0 else 0
            elif strategy == 'momentum':
                # 短线策略：30%资金 / 短线股票数量
                short_term_stocks = strategy_counts.get('momentum', 0)
                weight = self.short_term_ratio / short_term_stocks if short_term_stocks > 0 else 0
            else:
                weight = 0

            weights.append(weight)

        portfolio['weight'] = weights

        # 归一化（确保权重和为1）
        total_weight = portfolio['weight'].sum()
        if total_weight > 0:
            portfolio['weight'] = portfolio['weight'] / total_weight

        return portfolio

    def _check_industry_concentration(self, portfolio: pd.DataFrame):
        """检查行业集中度

        Args:
            portfolio: 组合DataFrame
        """
        if 'industry' not in portfolio.columns:
            return

        industry_weights = portfolio.groupby('industry')['weight'].sum().sort_values(ascending=False)

        print(f"\n行业分布:")
        for industry, weight in industry_weights.items():
            print(f"  {industry}: {weight:.1%}")

        # 警告：单行业占比过高
        max_concentration = industry_weights.max()
        if max_concentration > self.max_industry_ratio:
            print(f"\n⚠️  警告: {industry_weights.idxmax()} 行业占比 {max_concentration:.1%} "
                  f"超过阈值 {self.max_industry_ratio:.1%}")

    def rebalance(
        self,
        current_portfolio: pd.DataFrame,
        new_candidates: Dict[str, pd.DataFrame],
        rebalance_type: str = 'full',
    ) -> pd.DataFrame:
        """调仓逻辑

        Args:
            current_portfolio: 当前持仓
            new_candidates: 新的候选池 {'value': df, 'growth': df, 'momentum': df}
            rebalance_type: 调仓类型
                - 'full': 全部调仓（季度调仓）
                - 'short_only': 仅调仓短线（周度调仓）
                - 'long_only': 仅调仓长线（Regime切换时）

        Returns:
            新的投资组合
        """
        if rebalance_type == 'full':
            # 全部重新构建
            return self.construct_portfolio(
                value_candidates=new_candidates.get('value', pd.DataFrame()),
                growth_candidates=new_candidates.get('growth', pd.DataFrame()),
                momentum_candidates=new_candidates.get('momentum', pd.DataFrame()),
                total_positions=len(current_portfolio),
            )

        elif rebalance_type == 'short_only':
            # 保留长线持仓，只调仓短线
            long_term_holdings = current_portfolio[
                current_portfolio['strategy_type'].isin(['value', 'growth'])
            ].copy()

            # 重新选短线股票
            short_term_positions = len(current_portfolio) - len(long_term_holdings)
            momentum_candidates = new_candidates.get('momentum', pd.DataFrame())

            if not momentum_candidates.empty:
                new_momentum = self._select_from_pool(
                    momentum_candidates,
                    n_stocks=short_term_positions,
                    strategy_type='momentum',
                )
                portfolio = pd.concat([long_term_holdings, new_momentum], ignore_index=True)
            else:
                portfolio = long_term_holdings

            # 重新计算权重
            portfolio = self._calculate_weights(portfolio)
            return portfolio

        elif rebalance_type == 'long_only':
            # 保留短线持仓，只调仓长线
            short_term_holdings = current_portfolio[
                current_portfolio['strategy_type'] == 'momentum'
            ].copy()

            # 重新选长线股票
            long_term_positions = len(current_portfolio) - len(short_term_holdings)
            value_positions = int(long_term_positions * self.value_growth_ratio)
            growth_positions = long_term_positions - value_positions

            portfolio_parts = [short_term_holdings]

            value_candidates = new_candidates.get('value', pd.DataFrame())
            if not value_candidates.empty:
                new_value = self._select_from_pool(
                    value_candidates,
                    n_stocks=value_positions,
                    strategy_type='value',
                )
                portfolio_parts.append(new_value)

            growth_candidates = new_candidates.get('growth', pd.DataFrame())
            if not growth_candidates.empty:
                new_growth = self._select_from_pool(
                    growth_candidates,
                    n_stocks=growth_positions,
                    strategy_type='growth',
                )
                portfolio_parts.append(new_growth)

            portfolio = pd.concat(portfolio_parts, ignore_index=True)

            # 重新计算权重
            portfolio = self._calculate_weights(portfolio)
            return portfolio

        else:
            raise ValueError(f"不支持的调仓类型: {rebalance_type}")

    def apply_risk_control(
        self,
        portfolio: pd.DataFrame,
        current_drawdown: float,
    ) -> pd.DataFrame:
        """应用风险控制

        回撤控制规则：
        - 回撤 < -15%: 减仓50%
        - 回撤 < -25%: 清仓

        Args:
            portfolio: 当前组合
            current_drawdown: 当前回撤（负数）

        Returns:
            调整后的组合
        """
        portfolio = portfolio.copy()

        if current_drawdown <= -0.25:
            # 清仓
            print(f"⚠️  回撤 {current_drawdown:.1%} 触发清仓")
            portfolio['weight'] = 0
        elif current_drawdown <= -0.15:
            # 减仓50%
            print(f"⚠️  回撤 {current_drawdown:.1%} 触发减仓50%")
            portfolio['weight'] = portfolio['weight'] * 0.5
        else:
            # 正常持仓
            pass

        return portfolio
