#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
成长股规则选股器

选股逻辑：
1. 硬规则过滤：剔除不合格股票
2. 特征计算：计算成长股关键指标
3. 综合评分：多维度打分排序
4. 组合构建：行业分散 + 流动性约束
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class GrowthStockSelector:
    """成长股规则选股器

    核心理念：寻找"未来的巨头"
    - 高增长（营收CAGR > 20%）
    - 重研发（研发费用率 > 5%）
    - 强定价权（毛利率上升）
    - 高效率（资产周转率高）
    - 机构认可（基金持仓 > 20家）
    """

    def __init__(
        self,
        data_root: Path,
        min_revenue_cagr: float = 0.20,
        min_rd_ratio: float = 0.05,
        max_debt_ratio: float = 0.60,
        min_fund_holders: int = 20,
    ):
        """初始化成长股选股器

        Args:
            data_root: 数据根目录
            min_revenue_cagr: 最低营收CAGR（默认20%）
            min_rd_ratio: 最低研发费用率（默认5%）
            max_debt_ratio: 最高负债率（默认60%）
            min_fund_holders: 最少基金持仓家数（默认20家）
        """
        self.data_root = Path(data_root)
        self.min_revenue_cagr = min_revenue_cagr
        self.min_rd_ratio = min_rd_ratio
        self.max_debt_ratio = max_debt_ratio
        self.min_fund_holders = min_fund_holders

    def hard_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """硬规则过滤：不可妥协的底线

        Args:
            df: 包含所有特征的DataFrame

        Returns:
            通过硬规则的股票
        """
        filtered = df.copy()

        # 1. 营收CAGR > 20%（高增长底线）
        filtered = filtered[filtered["revenue_cagr_3y"] > self.min_revenue_cagr]

        # 2. 研发费用率 > 5%（创新投入底线）
        filtered = filtered[filtered["rd_ratio"] > self.min_rd_ratio]

        # 3. 负债率 < 60%（财务安全底线）
        filtered = filtered[filtered["debt_ratio"] < self.max_debt_ratio]

        # 4. 非ST股
        if "is_st" in filtered.columns:
            filtered = filtered[not filtered["is_st"]]

        # 5. 非退市股
        if "is_delisted" in filtered.columns:
            filtered = filtered[not filtered["is_delisted"]]

        # 6. 利润正增长（盈利能力底线）
        if "profit_cagr_3y" in filtered.columns:
            filtered = filtered[filtered["profit_cagr_3y"] > 0]

        return filtered

    def calculate_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算综合评分

        评分维度：
        1. 增长速度（35%）：营收CAGR + 利润CAGR
        2. 增长质量（25%）：研发费用率 + 毛利率趋势
        3. 运营效率（20%）：资产周转率 + ROE
        4. 行业地位（10%）：行业排名
        5. 机构认可（10%）：基金持仓 + 机构增持

        Args:
            df: 通过硬规则的股票

        Returns:
            带有评分的DataFrame
        """
        scored = df.copy()
        scored["score"] = 0.0

        # 1. 增长速度（35%）
        if "revenue_cagr_3y" in scored.columns:
            # 营收CAGR标准化（20%-50%映射到0-100）
            revenue_growth_score = (scored["revenue_cagr_3y"] - 0.20) / 0.30 * 100
            revenue_growth_score = revenue_growth_score.clip(0, 100)
            scored["score"] += revenue_growth_score * 0.20

        if "profit_cagr_3y" in scored.columns:
            # 利润CAGR标准化（0%-50%映射到0-100）
            profit_growth_score = (scored["profit_cagr_3y"]) / 0.50 * 100
            profit_growth_score = profit_growth_score.clip(0, 100)
            scored["score"] += profit_growth_score * 0.15

        # 2. 增长质量（25%）
        if "rd_ratio" in scored.columns:
            # 研发费用率（10%为满分）
            rd_score = (scored["rd_ratio"] / 0.10) * 100
            rd_score = rd_score.clip(0, 100)
            scored["score"] += rd_score * 0.15

        if "gross_margin_trend" in scored.columns:
            # 毛利率趋势（上升为正分）
            margin_trend_score = (scored["gross_margin_trend"] / 0.05) * 100
            margin_trend_score = margin_trend_score.clip(0, 100)
            scored["score"] += margin_trend_score * 0.10

        # 3. 运营效率（20%）
        if "asset_turnover" in scored.columns:
            # 资产周转率（2为满分）
            turnover_score = (scored["asset_turnover"] / 2.0) * 100
            turnover_score = turnover_score.clip(0, 100)
            scored["score"] += turnover_score * 0.10

        if "roe" in scored.columns:
            # ROE（30%为满分）
            roe_score = (scored["roe"] / 0.30) * 100
            roe_score = roe_score.clip(0, 100)
            scored["score"] += roe_score * 0.10

        # 4. 行业地位（10%）
        if "industry_rank" in scored.columns:
            # 行业排名（前3名为满分）
            rank_score = (4 - scored["industry_rank"]) / 3 * 100
            rank_score = rank_score.clip(0, 100)
            scored["score"] += rank_score * 0.10

        # 5. 机构认可（10%）
        if "fund_holders" in scored.columns:
            # 基金持仓家数（50家+为满分）
            fund_score = (scored["fund_holders"] / 50) * 100
            fund_score = fund_score.clip(0, 100)
            scored["score"] += fund_score * 0.05

        if "inst_holding_change" in scored.columns:
            # 机构增持（>10%为满分）
            inst_change_score = (scored["inst_holding_change"] / 0.10) * 100
            inst_change_score = inst_change_score.clip(0, 100)
            scored["score"] += inst_change_score * 0.05

        return scored.sort_values("score", ascending=False)

    def construct_portfolio(
        self,
        scored_df: pd.DataFrame,
        n_stocks: int = 5,
        max_industry_ratio: float = 0.40,
        min_avg_amount: float = 1e8,
    ) -> pd.DataFrame:
        """构建投资组合

        约束条件：
        1. 行业分散：单行业不超过40%（成长股允许更集中）
        2. 流动性：日均成交额 > 1亿
        3. 等权重配置

        Args:
            scored_df: 带有评分的股票
            n_stocks: 持仓数量（默认5只）
            max_industry_ratio: 单行业最大占比（默认40%）
            min_avg_amount: 最小日均成交额（默认1亿）

        Returns:
            投资组合DataFrame
        """
        portfolio = []
        industry_count = {}
        max_per_industry = int(n_stocks * max_industry_ratio)

        for _, stock in scored_df.iterrows():
            # 流动性约束
            if "avg_amount" in stock and stock["avg_amount"] < min_avg_amount:
                continue

            # 行业分散约束
            industry = stock.get("industry", "Unknown")
            if industry_count.get(industry, 0) >= max_per_industry:
                continue

            portfolio.append(stock)
            industry_count[industry] = industry_count.get(industry, 0) + 1

            if len(portfolio) >= n_stocks:
                break

        portfolio_df = pd.DataFrame(portfolio)

        # 等权重配置
        if not portfolio_df.empty:
            portfolio_df["weight"] = 1.0 / len(portfolio_df)

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
