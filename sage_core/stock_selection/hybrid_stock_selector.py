#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
混合选股器（硬规则 + 线性模型）

架构：
1. 第一层：硬规则过滤（剔除不合格股票）
2. 第二层：线性模型评分（学习特征权重）
3. 第三层：组合构建（行业分散 + 流动性约束）
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


class HybridStockSelector:
    """混合选股器（价值股/成长股通用）

    设计理念：
    - 硬规则保证基本面安全
    - 线性模型自动学习特征权重
    - 支持滚动训练（每季度重训练）
    """

    def __init__(
        self,
        selector_type: str,  # 'value' or 'growth'
        data_root: Path,
        model_path: Optional[Path] = None,
    ):
        """初始化混合选股器

        Args:
            selector_type: 选股器类型（'value' 或 'growth'）
            data_root: 数据根目录
            model_path: 模型保存路径（可选）
        """
        self.selector_type = selector_type
        self.data_root = Path(data_root)
        self.model_path = model_path
        self.model: Optional[LinearRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []

        # 根据类型设置硬规则和特征
        if selector_type == "value":
            self._setup_value_config()
        elif selector_type == "growth":
            self._setup_growth_config()
        else:
            raise ValueError(f"不支持的选股器类型: {selector_type}")

    def _setup_value_config(self):
        """配置价值股选股器"""
        # 硬规则阈值
        self.hard_rules = {
            "roe": (0.10, None),  # ROE > 10%
            "debt_ratio": (None, 0.70),  # 负债率 < 70%
            "consecutive_dividend": (3, None),  # 连续分红3年+
            "is_st": (False,),  # 非ST
        }

        # 模型特征（用于训练和预测）
        self.feature_names = [
            "roe",  # 盈利能力
            "roe_5y_avg",  # 盈利稳定性
            "debt_ratio",  # 财务安全
            "interest_coverage",  # 利息保障
            "consecutive_dividend",  # 分红能力
            "dividend_yield",  # 股息率
            "pe_relative",  # 估值水平
            "fund_holders",  # 机构认可
            "inst_holding_change",  # 机构增持
            "revenue_growth",  # 营收增长
        ]

    def _setup_growth_config(self):
        """配置成长股选股器"""
        # 硬规则阈值
        self.hard_rules = {
            "revenue_cagr_3y": (0.15, None),  # 营收CAGR > 15%
            "rd_ratio": (0.03, None),  # 研发费用率 > 3%
            "debt_ratio": (None, 0.60),  # 负债率 < 60%
            "is_st": (False,),  # 非ST
        }

        # 模型特征
        self.feature_names = [
            "revenue_cagr_3y",  # 营收增长
            "profit_cagr_3y",  # 利润增长
            "rd_ratio",  # 研发投入
            "gross_margin",  # 毛利率
            "gross_margin_trend",  # 毛利率趋势
            "asset_turnover",  # 运营效率
            "roe",  # 盈利能力
            "industry_rank",  # 行业地位
            "fund_holders",  # 机构认可
            "inst_holding_change",  # 机构增持
        ]

    def hard_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """硬规则过滤

        Args:
            df: 包含所有特征的DataFrame

        Returns:
            通过硬规则的股票
        """
        filtered = df.copy()

        for feature, threshold in self.hard_rules.items():
            if feature not in filtered.columns:
                continue

            if len(threshold) == 1:
                # 布尔值判断
                filtered = filtered[filtered[feature] == threshold[0]]
            elif threshold[0] is not None and threshold[1] is not None:
                # 区间判断
                filtered = filtered[(filtered[feature] >= threshold[0]) & (filtered[feature] <= threshold[1])]
            elif threshold[0] is not None:
                # 下限判断
                filtered = filtered[filtered[feature] >= threshold[0]]
            elif threshold[1] is not None:
                # 上限判断
                filtered = filtered[filtered[feature] <= threshold[1]]

        return filtered

    def train(
        self,
        df_train: pd.DataFrame,
        target_col: str = "return_6m",
    ) -> Dict[str, float]:
        """训练线性模型

        Args:
            df_train: 训练数据（包含特征和标签）
            target_col: 目标列名（默认未来6个月收益率）

        Returns:
            训练指标字典
        """
        # 1. 硬规则过滤
        df_filtered = self.hard_filter(df_train)
        print(f"硬规则过滤: {len(df_train)} -> {len(df_filtered)} 只股票")

        if df_filtered.empty:
            raise ValueError("没有股票通过硬规则过滤")

        # 2. 准备特征和标签
        available_features = [f for f in self.feature_names if f in df_filtered.columns]
        if not available_features:
            raise ValueError(f"没有可用特征，需要: {self.feature_names}")

        X = df_filtered[available_features].copy()
        y = df_filtered[target_col].copy()

        # 删除缺失值
        valid_mask = X.notna().all(axis=1) & y.notna()
        X = X[valid_mask]
        y = y[valid_mask]

        print(f"训练样本数: {len(X)}, 特征数: {len(available_features)}")

        # 3. 标准化特征
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 4. 训练线性模型
        self.model = LinearRegression()
        self.model.fit(X_scaled, y)

        # 5. 计算训练指标
        y_pred = self.model.predict(X_scaled)
        r2 = self.model.score(X_scaled, y)
        mse = np.mean((y - y_pred) ** 2)

        # 6. 打印特征权重
        print("\n特征权重（标准化后）：")
        feature_importance = pd.DataFrame({"feature": available_features, "weight": self.model.coef_}).sort_values(
            "weight", ascending=False
        )
        print(feature_importance.to_string(index=False))

        # 7. 保存模型
        if self.model_path:
            self._save_model()

        return {
            "r2": r2,
            "mse": mse,
            "n_samples": len(X),
            "n_features": len(available_features),
        }

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """预测股票评分

        Args:
            df: 包含所有特征的DataFrame

        Returns:
            带有评分的DataFrame
        """
        if self.model is None or self.scaler is None:
            raise ValueError("模型未训练，请先调用train()或load_model()")

        # 1. 硬规则过滤
        df_filtered = self.hard_filter(df)
        print(f"硬规则过滤: {len(df)} -> {len(df_filtered)} 只股票")

        if df_filtered.empty:
            print("警告: 没有股票通过硬规则过滤")
            return pd.DataFrame()

        # 2. 准备特征
        available_features = [f for f in self.feature_names if f in df_filtered.columns]
        X = df_filtered[available_features].copy()

        # 删除缺失值
        valid_mask = X.notna().all(axis=1)
        df_valid = df_filtered[valid_mask].copy()
        X_valid = X[valid_mask]

        if df_valid.empty:
            print("警告: 所有股票都有缺失特征")
            return pd.DataFrame()

        # 3. 标准化并预测
        X_scaled = self.scaler.transform(X_valid)
        df_valid["score"] = self.model.predict(X_scaled)

        # 4. 按评分排序
        df_valid = df_valid.sort_values("score", ascending=False)

        return df_valid

    def select(
        self,
        df: pd.DataFrame,
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """执行选股流程（仅选股+评分，不构建组合）

        Args:
            df: 包含所有特征的股票池
            top_n: 返回前N只股票（可选，None表示返回所有通过硬规则的股票）

        Returns:
            带有评分的候选股票池
        """
        # 第一层：硬规则过滤 + 第二层：模型评分
        scored = self.predict(df)

        if scored.empty:
            return pd.DataFrame()

        print(f"选股完成: {len(scored)} 只候选股票")
        print(f"  最高分: {scored['score'].max():.4f}")
        print(f"  最低分: {scored['score'].min():.4f}")
        print(f"  平均分: {scored['score'].mean():.4f}")

        # 可选：只返回前N只
        if top_n is not None:
            scored = scored.head(top_n)

        return scored

    def _save_model(self):
        """保存模型到文件"""
        if self.model_path is None:
            return

        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "selector_type": self.selector_type,
            "hard_rules": self.hard_rules,
        }

        with open(self.model_path, "wb") as f:
            pickle.dump(model_data, f)

        print(f"模型已保存: {self.model_path}")

    def load_model(self, model_path: Optional[Path] = None):
        """从文件加载模型

        Args:
            model_path: 模型文件路径（可选，默认使用初始化时的路径）
        """
        path = model_path or self.model_path
        if path is None:
            raise ValueError("未指定模型路径")

        with open(path, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.feature_names = model_data["feature_names"]
        self.selector_type = model_data["selector_type"]
        self.hard_rules = model_data["hard_rules"]

        print(f"模型已加载: {path}")
