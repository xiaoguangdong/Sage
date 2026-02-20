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
    - 高效率（资产周转率高 + 费用率合理）
    - 现金流真实（CFO/净利润健康）
    - 机构认可（基金持仓 > 20家）
    """

    def __init__(
        self,
        data_root: Path,
        rule_config: dict | None = None,
    ):
        """初始化成长股选股器

        Args:
            data_root: 数据根目录
            rule_config: 规则配置（硬规则/评分权重/评分参数）
        """
        self.data_root = Path(data_root)
        self.rule_config = rule_config or {}
        self.hard_filters = self.rule_config.get("hard_filters", {})
        self.score_weights = self.rule_config.get("score_weights", {})
        self.score_params = self.rule_config.get("score_params", {})

    @staticmethod
    def _normalize_score(series: pd.Series, params: dict) -> pd.Series:
        min_val = params.get("min")
        max_val = params.get("max")
        higher_better = params.get("higher_better", True)
        if min_val is None or max_val is None or max_val == min_val:
            return pd.Series(index=series.index, data=pd.NA)
        if higher_better:
            score = (series - min_val) / (max_val - min_val)
        else:
            score = (max_val - series) / (max_val - min_val)
        return (score.clip(0, 1) * 100).astype(float)

    def hard_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """硬规则过滤：不可妥协的底线

        Args:
            df: 包含所有特征的DataFrame

        Returns:
            通过硬规则的股票
        """
        filtered = df.copy()

        industry_cfg = self.rule_config.get("industry_quantile", {}) or {}
        if industry_cfg.get("enabled", False):
            filtered = self._apply_industry_quantile_rules(filtered, industry_cfg)

        for field, rule in self.hard_filters.items():
            if field not in filtered.columns:
                continue
            min_val = rule.get("min")
            max_val = rule.get("max")
            eq_val = rule.get("eq")
            if eq_val is not None:
                filtered = filtered[filtered[field] == eq_val]
            else:
                if min_val is not None:
                    filtered = filtered[filtered[field] >= min_val]
                if max_val is not None:
                    filtered = filtered[filtered[field] <= max_val]

        # 4. 非ST股
        if "is_st" in filtered.columns:
            filtered = filtered[~filtered["is_st"]]

        # 5. 非退市股
        if "is_delisted" in filtered.columns:
            filtered = filtered[~filtered["is_delisted"]]

        # 6. 利润正增长（盈利能力底线）
        if "profit_cagr_3y" in filtered.columns and self.hard_filters.get("profit_cagr_3y", {}) != {}:
            rule = self.hard_filters.get("profit_cagr_3y", {})
            min_val = rule.get("min")
            if min_val is not None:
                filtered = filtered[filtered["profit_cagr_3y"] >= min_val]

        return filtered

    def _apply_industry_quantile_rules(self, df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
        industry_col = cfg.get("industry_col", "industry_l1")
        top_pct = float(cfg.get("top_pct", 0.30))
        min_samples = int(cfg.get("min_samples", 30))
        fallback = cfg.get("fallback", "market")
        rules = cfg.get("rules", {}) or {}

        if industry_col not in df.columns:
            if fallback != "market":
                return df
            industry_col = None

        filtered = df.copy()
        for feature, direction in rules.items():
            if feature not in filtered.columns:
                continue

            series = filtered[feature]
            if industry_col:
                quantiles = filtered.groupby(industry_col)[feature].transform(
                    lambda s: (
                        s.rank(pct=True)
                        if s.notna().sum() >= min_samples
                        else pd.Series([pd.NA] * len(s), index=s.index)
                    )
                )
                use_market = quantiles.isna()
                if fallback == "market" and use_market.any():
                    market_q = series.rank(pct=True)
                    quantiles = quantiles.fillna(market_q)
            else:
                quantiles = series.rank(pct=True)

            if direction == "top":
                filtered = filtered[quantiles >= (1 - top_pct)]
            elif direction == "bottom":
                filtered = filtered[quantiles <= top_pct]

        return filtered

    def calculate_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算综合评分

        评分维度：
        1. 增长速度（30%）：营收CAGR + 利润CAGR
        2. 增长质量（25%）：研发费用率 + 毛利率趋势 + 真实利润
        3. 运营效率（20%）：资产周转率 + ROE + 费用率
        4. 行业地位（10%）：行业排名
        5. 机构认可（10%）：基金持仓 + 机构增持
        6. 财务安全（5%）：净现金水平

        Args:
            df: 通过硬规则的股票

        Returns:
            带有评分的DataFrame
        """
        scored = df.copy()
        scored["score"] = 0.0

        for feature, weight in self.score_weights.items():
            if feature not in scored.columns:
                continue
            params = self.score_params.get(feature, {})
            score_series = self._normalize_score(scored[feature], params)
            if score_series.isna().all():
                continue
            scored["score"] += score_series.fillna(0) * float(weight)

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
