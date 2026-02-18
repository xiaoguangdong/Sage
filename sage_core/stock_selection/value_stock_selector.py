#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
价值股规则选股器

选股逻辑：
1. 硬规则过滤：剔除不合格股票
2. 特征计算：计算价值股关键指标
3. 综合评分：多维度打分排序
4. 组合构建：行业分散 + 流动性约束
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path


class ValueStockSelector:
    """价值股规则选股器

    核心理念：寻找"便宜的好公司"
    - 盈利能力强（ROE > 15%）
    - 财务安全（负债率 < 60%）
    - 现金流稳定（连续分红5年+）
    - 估值合理（PE < 行业中位数）
    - 机构认可（基金持仓 > 20家）
    """

    def __init__(
        self,
        data_root: Path,
        min_roe: float = 0.15,
        max_debt_ratio: float = 0.60,
        min_consecutive_dividend: int = 5,
        min_fund_holders: int = 20,
    ):
        """初始化价值股选股器

        Args:
            data_root: 数据根目录
            min_roe: 最低ROE要求（默认15%）
            max_debt_ratio: 最高负债率（默认60%）
            min_consecutive_dividend: 最少连续分红年数（默认5年）
            min_fund_holders: 最少基金持仓家数（默认20家）
        """
        self.data_root = Path(data_root)
        self.min_roe = min_roe
        self.max_debt_ratio = max_debt_ratio
        self.min_consecutive_dividend = min_consecutive_dividend
        self.min_fund_holders = min_fund_holders

    def hard_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """硬规则过滤：不可妥协的底线

        Args:
            df: 包含所有特征的DataFrame

        Returns:
            通过硬规则的股票
        """
        filtered = df.copy()

        # 1. ROE > 15%（盈利能力底线）
        filtered = filtered[filtered['roe'] > self.min_roe]

        # 2. 负债率 < 60%（财务安全底线）
        filtered = filtered[filtered['debt_ratio'] < self.max_debt_ratio]

        # 3. 连续分红5年+（现金流稳定）
        filtered = filtered[filtered['consecutive_dividend'] >= self.min_consecutive_dividend]

        # 4. 非ST股
        if 'is_st' in filtered.columns:
            filtered = filtered[filtered['is_st'] == False]

        # 5. 非退市股
        if 'is_delisted' in filtered.columns:
            filtered = filtered[filtered['is_delisted'] == False]

        # 6. 营收正增长（成长性底线）
        if 'revenue_growth' in filtered.columns:
            filtered = filtered[filtered['revenue_growth'] > 0]

        return filtered

    def calculate_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算综合评分

        评分维度：
        1. 盈利能力（30%）：ROE + ROE稳定性
        2. 财务安全（25%）：负债率 + 利息保障倍数
        3. 分红能力（20%）：连续分红年数 + 股息率
        4. 估值水平（15%）：PE相对值
        5. 机构认可（10%）：基金持仓 + 机构增持

        Args:
            df: 通过硬规则的股票

        Returns:
            带有评分的DataFrame
        """
        scored = df.copy()
        scored['score'] = 0.0

        # 1. 盈利能力（30%）
        if 'roe' in scored.columns:
            # ROE标准化（0-100分）
            roe_score = (scored['roe'] - 0.10) / 0.30 * 100  # 10%-40%映射到0-100
            roe_score = roe_score.clip(0, 100)
            scored['score'] += roe_score * 0.20

        if 'roe_5y_std' in scored.columns:
            # ROE稳定性（标准差越小越好）
            roe_stability_score = (1 - scored['roe_5y_std'] / 0.10) * 100
            roe_stability_score = roe_stability_score.clip(0, 100)
            scored['score'] += roe_stability_score * 0.10

        # 2. 财务安全（25%）
        if 'debt_ratio' in scored.columns:
            # 负债率（越低越好）
            debt_score = (1 - scored['debt_ratio']) * 100
            debt_score = debt_score.clip(0, 100)
            scored['score'] += debt_score * 0.15

        if 'interest_coverage' in scored.columns:
            # 利息保障倍数（>5倍为优秀）
            interest_score = (scored['interest_coverage'] / 10) * 100
            interest_score = interest_score.clip(0, 100)
            scored['score'] += interest_score * 0.10

        # 3. 分红能力（20%）
        if 'consecutive_dividend' in scored.columns:
            # 连续分红年数（10年+为满分）
            dividend_years_score = (scored['consecutive_dividend'] / 10) * 100
            dividend_years_score = dividend_years_score.clip(0, 100)
            scored['score'] += dividend_years_score * 0.10

        if 'dividend_yield' in scored.columns:
            # 股息率（5%为满分）
            dividend_yield_score = (scored['dividend_yield'] / 0.05) * 100
            dividend_yield_score = dividend_yield_score.clip(0, 100)
            scored['score'] += dividend_yield_score * 0.10

        # 4. 估值水平（15%）
        if 'pe_relative' in scored.columns:
            # PE相对值（<0.8为优秀）
            pe_score = (1 - scored['pe_relative']) * 100
            pe_score = pe_score.clip(0, 100)
            scored['score'] += pe_score * 0.15

        # 5. 机构认可（10%）
        if 'fund_holders' in scored.columns:
            # 基金持仓家数（50家+为满分）
            fund_score = (scored['fund_holders'] / 50) * 100
            fund_score = fund_score.clip(0, 100)
            scored['score'] += fund_score * 0.05

        if 'inst_holding_change' in scored.columns:
            # 机构增持（>10%为满分）
            inst_change_score = (scored['inst_holding_change'] / 0.10) * 100
            inst_change_score = inst_change_score.clip(0, 100)
            scored['score'] += inst_change_score * 0.05

        return scored.sort_values('score', ascending=False)

    def construct_portfolio(
        self,
        scored_df: pd.DataFrame,
        n_stocks: int = 5,
        max_industry_ratio: float = 0.30,
        min_avg_amount: float = 1e8,
    ) -> pd.DataFrame:
        """构建投资组合

        约束条件：
        1. 行业分散：单行业不超过30%
        2. 流动性：日均成交额 > 1亿
        3. 等权重配置

        Args:
            scored_df: 带有评分的股票
            n_stocks: 持仓数量（默认5只）
            max_industry_ratio: 单行业最大占比（默认30%）
            min_avg_amount: 最小日均成交额（默认1亿）

        Returns:
            投资组合DataFrame
        """
        portfolio = []
        industry_count = {}
        max_per_industry = int(n_stocks * max_industry_ratio)

        for _, stock in scored_df.iterrows():
            # 流动性约束
            if 'avg_amount' in stock and stock['avg_amount'] < min_avg_amount:
                continue

            # 行业分散约束
            industry = stock.get('industry', 'Unknown')
            if industry_count.get(industry, 0) >= max_per_industry:
                continue

            portfolio.append(stock)
            industry_count[industry] = industry_count.get(industry, 0) + 1

            if len(portfolio) >= n_stocks:
                break

        portfolio_df = pd.DataFrame(portfolio)

        # 等权重配置
        if not portfolio_df.empty:
            portfolio_df['weight'] = 1.0 / len(portfolio_df)

        return portfolio_df

    def select(
        self,
        df: pd.DataFrame,
        n_stocks: int = 5,
    ) -> pd.DataFrame:
        """执行选股流程

        Args:
            df: 包含所有特征的股票池
            n_stocks: 目标持仓数量

        Returns:
            最终投资组合
        """
        # 第一层：硬规则过滤
        filtered = self.hard_filter(df)
        print(f"硬规则过滤: {len(df)} -> {len(filtered)} 只股票")

        if filtered.empty:
            print("警告: 没有股票通过硬规则过滤")
            return pd.DataFrame()

        # 第二层：综合评分
        scored = self.calculate_score(filtered)
        print(f"评分完成，最高分: {scored['score'].max():.2f}")

        # 第三层：组合构建
        portfolio = self.construct_portfolio(scored, n_stocks=n_stocks)
        print(f"组合构建完成: {len(portfolio)} 只股票")

        return portfolio
