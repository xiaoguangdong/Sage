#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
组合构建优化

本模块提供三大组合优化功能，用于构建更稳健的投资组合：

1. **IC 加权持仓**（IC-Weighted Portfolio）
   - 目的：根据因子得分分配仓位，而非等权
   - 方法：
     * softmax: exp(score/T) 加权，温度参数控制集中度
     * linear: score 线性加权
     * rank: 排名加权
   - 适用场景：当因子得分有明确预测能力时

2. **换手率约束**（Turnover Constraint）
   - 目的：减少不必要的调仓，降低交易成本
   - 方法：限制单次调仓的权重变化总和
   - 适用场景：高频调仓策略、交易成本敏感场景

3. **行业暴露约束**（Sector Exposure Constraint）
   - 目的：更严格的行业分散，避免行业集中风险
   - 方法：限制单行业最大权重
   - 适用场景：需要严格风控的组合

使用示例：
    >>> from sage_core.portfolio.portfolio_optimizer import PortfolioOptimizer
    >>> optimizer = PortfolioOptimizer(
    ...     max_turnover=0.30,
    ...     max_industry_weight=0.30
    ... )
    >>>
    >>> # IC 加权持仓
    >>> weighted_df = optimizer.ic_weighted_portfolio(
    ...     df,
    ...     score_col='score',
    ...     top_n=20,
    ...     weight_method='softmax',
    ...     temperature=1.0
    ... )
    >>>
    >>> # 换手率约束
    >>> constrained_df = optimizer.apply_turnover_constraint(
    ...     new_weights_df,
    ...     old_weights,
    ...     max_turnover=0.30
    ... )
    >>>
    >>> # 行业暴露约束
    >>> sector_constrained_df = optimizer.apply_sector_constraint(
    ...     weights_df,
    ...     max_exposure=0.30
    ... )

权重方法对比：
    - softmax: 适合得分差异明显的场景，温度参数控制集中度
      * temperature=0.5: 高集中度（头部股票权重大）
      * temperature=1.0: 中等集中度（默认）
      * temperature=2.0: 低集中度（接近等权）
    - linear: 适合得分线性相关的场景
    - rank: 适合只关心排名的场景

注意事项：
    - IC 加权会创建新列 'weight'
    - 换手率约束可能导致部分股票无法调整
    - 行业约束可能导致部分高分股票被剔除
    - 所有方法都会自动归一化权重（总和为1）
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


class PortfolioOptimizer:
    """组合构建优化器"""

    def __init__(
        self,
        max_turnover: float = 0.30,
        max_industry_weight: float = 0.30,
        min_position_weight: float = 0.01,
    ):
        """初始化

        Args:
            max_turnover: 最大单次换手率（0.30 表示 30%）
            max_industry_weight: 单行业最大权重（0.30 表示 30%）
            min_position_weight: 最小持仓权重（低于此值的持仓会被清除）
        """
        self.max_turnover = max_turnover
        self.max_industry_weight = max_industry_weight
        self.min_position_weight = min_position_weight

    # ==================== IC 加权持仓 ====================

    def ic_weighted_portfolio(
        self,
        df: pd.DataFrame,
        score_col: str = "score",
        top_n: Optional[int] = None,
        weight_method: str = "softmax",
        temperature: float = 1.0,
    ) -> pd.DataFrame:
        """IC 加权持仓（按 score 加权分配仓位）

        Args:
            df: 包含 score 的 DataFrame
            score_col: 评分列名
            top_n: 只选择前 N 只股票（None 表示全部）
            weight_method: 权重计算方法（'softmax', 'linear', 'rank'）
            temperature: softmax 温度参数（越小权重越集中）

        Returns:
            添加 weight 列的 DataFrame
        """
        result = df.copy()

        if score_col not in result.columns:
            raise ValueError(f"缺少评分列: {score_col}")

        # 只保留有效评分
        result = result[result[score_col].notna()].copy()

        if result.empty:
            return result

        # 按评分排序
        result = result.sort_values(score_col, ascending=False)

        # 只选择前 N 只
        if top_n is not None:
            result = result.head(top_n)

        # 计算权重
        scores = result[score_col].values

        if weight_method == "softmax":
            # Softmax 权重（评分越高权重越大）
            exp_scores = np.exp(scores / temperature)
            weights = exp_scores / exp_scores.sum()

        elif weight_method == "linear":
            # 线性权重（评分归一化）
            min_score = scores.min()
            max_score = scores.max()
            if max_score > min_score:
                normalized_scores = (scores - min_score) / (max_score - min_score)
            else:
                normalized_scores = np.ones_like(scores)
            weights = normalized_scores / normalized_scores.sum()

        elif weight_method == "rank":
            # 排名权重（排名越靠前权重越大）
            ranks = np.arange(len(scores), 0, -1)
            weights = ranks / ranks.sum()

        else:
            raise ValueError(f"不支持的权重方法: {weight_method}")

        result["weight"] = weights

        return result

    # ==================== 换手率约束 ====================

    def turnover_constrained_rebalance(
        self,
        new_portfolio: pd.DataFrame,
        old_portfolio: Optional[pd.DataFrame] = None,
        ts_code_col: str = "ts_code",
        weight_col: str = "weight",
    ) -> pd.DataFrame:
        """换手率约束调仓

        保留上期持仓，只调整差异部分，控制换手率

        Args:
            new_portfolio: 新组合（包含 ts_code 和 weight）
            old_portfolio: 旧组合（包含 ts_code 和 weight），None 表示首次建仓
            ts_code_col: 股票代码列名
            weight_col: 权重列名

        Returns:
            调整后的组合
        """
        if old_portfolio is None or old_portfolio.empty:
            # 首次建仓，无需约束
            return new_portfolio

        # 合并新旧组合
        old_weights = old_portfolio.set_index(ts_code_col)[weight_col].to_dict()
        new_weights = new_portfolio.set_index(ts_code_col)[weight_col].to_dict()

        # 计算换手率
        all_codes = set(old_weights.keys()) | set(new_weights.keys())
        turnover = 0.0
        for code in all_codes:
            old_w = old_weights.get(code, 0.0)
            new_w = new_weights.get(code, 0.0)
            turnover += abs(new_w - old_w)

        turnover = turnover / 2  # 单边换手率

        if turnover <= self.max_turnover:
            # 换手率在限制内，直接返回新组合
            return new_portfolio

        # 换手率超限，需要调整
        print(f"换手率 {turnover:.2%} 超过限制 {self.max_turnover:.2%}，进行调整")

        # 计算调整系数
        adjustment_factor = self.max_turnover / turnover

        # 调整权重：保留部分旧权重
        adjusted_weights = {}
        for code in all_codes:
            old_w = old_weights.get(code, 0.0)
            new_w = new_weights.get(code, 0.0)
            # 线性插值
            adjusted_w = old_w + (new_w - old_w) * adjustment_factor
            if adjusted_w >= self.min_position_weight:
                adjusted_weights[code] = adjusted_w

        # 归一化权重
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}

        # 构建调整后的组合
        result = pd.DataFrame([{ts_code_col: code, weight_col: weight} for code, weight in adjusted_weights.items()])

        # 合并其他列（从新组合中）
        other_cols = [col for col in new_portfolio.columns if col not in [ts_code_col, weight_col]]
        if other_cols:
            result = result.merge(
                new_portfolio[[ts_code_col] + other_cols],
                on=ts_code_col,
                how="left",
            )

        return result

    # ==================== 行业暴露约束 ====================

    def industry_exposure_constraint(
        self,
        portfolio: pd.DataFrame,
        industry_col: str = "industry",
        ts_code_col: str = "ts_code",
        weight_col: str = "weight",
    ) -> pd.DataFrame:
        """行业暴露约束

        确保单行业权重不超过限制

        Args:
            portfolio: 组合（包含 ts_code, weight, industry）
            industry_col: 行业列名
            ts_code_col: 股票代码列名
            weight_col: 权重列名

        Returns:
            调整后的组合
        """
        result = portfolio.copy()

        if industry_col not in result.columns:
            print(f"警告: 缺少行业列 {industry_col}，跳过行业约束")
            return result

        # 计算行业权重
        industry_weights = result.groupby(industry_col)[weight_col].sum()

        # 检查是否有行业超限
        over_limit_industries = industry_weights[industry_weights > self.max_industry_weight].index.tolist()

        if not over_limit_industries:
            # 所有行业都在限制内
            return result

        print(f"行业超限: {over_limit_industries}，进行调整")

        # 调整超限行业的权重
        for industry in over_limit_industries:
            # 该行业的股票
            industry_mask = result[industry_col] == industry
            industry_stocks = result[industry_mask].copy()

            # 当前行业总权重
            current_weight = industry_stocks[weight_col].sum()

            # 缩放因子
            scale_factor = self.max_industry_weight / current_weight

            # 调整权重
            result.loc[industry_mask, weight_col] = result.loc[industry_mask, weight_col] * scale_factor

        # 重新归一化权重
        total_weight = result[weight_col].sum()
        if total_weight > 0:
            result[weight_col] = result[weight_col] / total_weight

        # 删除权重过小的持仓
        result = result[result[weight_col] >= self.min_position_weight].copy()

        # 再次归一化
        total_weight = result[weight_col].sum()
        if total_weight > 0:
            result[weight_col] = result[weight_col] / total_weight

        return result

    # ==================== 综合优化 ====================

    def optimize_portfolio(
        self,
        new_portfolio: pd.DataFrame,
        old_portfolio: Optional[pd.DataFrame] = None,
        score_col: str = "score",
        industry_col: str = "industry",
        ts_code_col: str = "ts_code",
        weight_method: str = "softmax",
        temperature: float = 1.0,
    ) -> pd.DataFrame:
        """综合优化组合

        依次应用：IC 加权 → 换手率约束 → 行业暴露约束

        Args:
            new_portfolio: 新组合（包含 score, industry）
            old_portfolio: 旧组合（包含 ts_code, weight）
            score_col: 评分列名
            industry_col: 行业列名
            ts_code_col: 股票代码列名
            weight_method: 权重计算方法
            temperature: softmax 温度参数

        Returns:
            优化后的组合
        """
        # 1. IC 加权持仓
        portfolio = self.ic_weighted_portfolio(
            new_portfolio,
            score_col=score_col,
            weight_method=weight_method,
            temperature=temperature,
        )

        # 2. 换手率约束
        portfolio = self.turnover_constrained_rebalance(
            portfolio,
            old_portfolio=old_portfolio,
            ts_code_col=ts_code_col,
            weight_col="weight",
        )

        # 3. 行业暴露约束
        portfolio = self.industry_exposure_constraint(
            portfolio,
            industry_col=industry_col,
            ts_code_col=ts_code_col,
            weight_col="weight",
        )

        return portfolio

    # ==================== 组合分析 ====================

    def analyze_portfolio(
        self,
        portfolio: pd.DataFrame,
        old_portfolio: Optional[pd.DataFrame] = None,
        industry_col: str = "industry",
        ts_code_col: str = "ts_code",
        weight_col: str = "weight",
    ) -> Dict:
        """分析组合特征

        Args:
            portfolio: 当前组合
            old_portfolio: 旧组合（用于计算换手率）
            industry_col: 行业列名
            ts_code_col: 股票代码列名
            weight_col: 权重列名

        Returns:
            分析结果字典
        """
        analysis = {}

        # 1. 持仓数量
        analysis["n_positions"] = len(portfolio)

        # 2. 权重分布
        analysis["max_weight"] = portfolio[weight_col].max()
        analysis["min_weight"] = portfolio[weight_col].min()
        analysis["mean_weight"] = portfolio[weight_col].mean()
        analysis["weight_std"] = portfolio[weight_col].std()

        # 3. 行业分布
        if industry_col in portfolio.columns:
            industry_weights = portfolio.groupby(industry_col)[weight_col].sum()
            analysis["n_industries"] = len(industry_weights)
            analysis["max_industry_weight"] = industry_weights.max()
            analysis["industry_concentration"] = (industry_weights**2).sum()  # HHI

        # 4. 换手率
        if old_portfolio is not None and not old_portfolio.empty:
            old_weights = old_portfolio.set_index(ts_code_col)[weight_col].to_dict()
            new_weights = portfolio.set_index(ts_code_col)[weight_col].to_dict()
            all_codes = set(old_weights.keys()) | set(new_weights.keys())
            turnover = sum(abs(new_weights.get(code, 0) - old_weights.get(code, 0)) for code in all_codes) / 2
            analysis["turnover"] = turnover

        return analysis
