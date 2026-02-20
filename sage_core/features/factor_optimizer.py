#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
因子处理优化

本模块提供三大因子优化功能，用于提升因子质量和降低过拟合风险：

1. **市值中性化**（Market Cap Neutralization）
   - 目的：消除市值因子对其他因子的影响
   - 方法：对每个因子做市值回归，取残差作为中性化后的因子
   - 适用场景：当因子与市值高度相关时（如成交额、机构持仓等）

2. **因子正交化**（Factor Orthogonalization）
   - 目的：消除因子间的共线性，提取独立信号
   - 方法：使用 PCA 或 Gram-Schmidt 正交化
   - 适用场景：当多个因子高度相关时（如不同周期的动量因子）

3. **多 Horizon 标签融合**（Multi-Horizon Label Fusion）
   - 目的：融合不同预测周期的标签，提升预测稳定性
   - 方法：加权平均多个周期的收益率
   - 适用场景：当需要平衡短期和长期预测时

使用示例：
    >>> from sage_core.features.factor_optimizer import FactorOptimizer
    >>> optimizer = FactorOptimizer()
    >>>
    >>> # 市值中性化
    >>> df_neutral = optimizer.market_cap_neutralization(
    ...     df,
    ...     factor_cols=['momentum_20d', 'volume_ratio'],
    ...     market_cap_col='total_mv',
    ...     n_bins=10
    ... )
    >>>
    >>> # 因子正交化
    >>> df_ortho = optimizer.factor_orthogonalization(
    ...     df,
    ...     factor_cols=['momentum_5d', 'momentum_10d', 'momentum_20d'],
    ...     method='pca',
    ...     n_components=2
    ... )
    >>>
    >>> # 多周期标签融合
    >>> df_fused = optimizer.multi_horizon_label_fusion(
    ...     df,
    ...     label_cols=['ret_5d', 'ret_10d', 'ret_20d'],
    ...     weights=[0.3, 0.3, 0.4]
    ... )

注意事项：
    - 市值中性化会创建新列（原列名 + _neutral 后缀）
    - 因子正交化会替换原因子列
    - 标签融合会创建新列 'label_fused'
    - 所有方法都会自动处理缺失值
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


class FactorOptimizer:
    """因子处理优化器"""

    def __init__(self):
        self.neutralization_models: Dict[str, LinearRegression] = {}
        self.orthogonalization_matrix: Optional[np.ndarray] = None
        self.feature_names: List[str] = []

    # ==================== 市值中性化 ====================

    def market_cap_neutralization(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        market_cap_col: str = "total_mv",
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """市值中性化处理

        对每个因子做市值分组回归，取残差作为中性化后的因子

        Args:
            df: 包含因子和市值的 DataFrame
            factor_cols: 需要中性化的因子列表
            market_cap_col: 市值列名
            n_bins: 市值分组数量

        Returns:
            中性化后的 DataFrame（原始列 + _neutral 后缀列）
        """
        result = df.copy()

        if market_cap_col not in df.columns:
            print(f"警告: 缺少市值列 {market_cap_col}，跳过市值中性化")
            return result

        # 市值分组（按百分位）
        result["mv_group"] = pd.qcut(result[market_cap_col], q=n_bins, labels=False, duplicates="drop")

        for factor in factor_cols:
            if factor not in df.columns:
                continue

            # 对每个因子做市值回归
            valid_mask = result[factor].notna() & result[market_cap_col].notna()
            if valid_mask.sum() < 10:
                print(f"警告: 因子 {factor} 有效样本不足，跳过中性化")
                continue

            X = result.loc[valid_mask, [market_cap_col]].values
            y = result.loc[valid_mask, factor].values

            # 线性回归
            model = LinearRegression()
            model.fit(X, y)

            # 残差作为中性化后的因子
            y_pred = model.predict(X)
            residuals = y - y_pred

            # 保存中性化后的因子
            result.loc[valid_mask, f"{factor}_neutral"] = residuals

            # 保存模型（用于预测时）
            self.neutralization_models[factor] = model

        # 删除临时列
        result = result.drop(columns=["mv_group"])

        return result

    def market_cap_neutralize(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        cap_col: str = "total_mv",
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """兼容旧接口：市值中性化"""
        return self.market_cap_neutralization(
            df=df,
            factor_cols=factor_cols,
            market_cap_col=cap_col,
            n_bins=n_bins,
        )

    def apply_market_cap_neutralization(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        market_cap_col: str = "total_mv",
    ) -> pd.DataFrame:
        """应用已训练的市值中性化模型

        Args:
            df: 包含因子和市值的 DataFrame
            factor_cols: 需要中性化的因子列表
            market_cap_col: 市值列名

        Returns:
            中性化后的 DataFrame
        """
        result = df.copy()

        if market_cap_col not in df.columns:
            return result

        for factor in factor_cols:
            if factor not in df.columns or factor not in self.neutralization_models:
                continue

            valid_mask = result[factor].notna() & result[market_cap_col].notna()
            if valid_mask.sum() == 0:
                continue

            X = result.loc[valid_mask, [market_cap_col]].values
            y = result.loc[valid_mask, factor].values

            # 使用已训练的模型预测
            model = self.neutralization_models[factor]
            y_pred = model.predict(X)
            residuals = y - y_pred

            result.loc[valid_mask, f"{factor}_neutral"] = residuals

        return result

    # ==================== 因子正交化 ====================

    def factor_orthogonalization(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        method: str = "pca",
        n_components: Optional[int] = None,
    ) -> pd.DataFrame:
        """因子正交化处理

        消除因子之间的共线性

        Args:
            df: 包含因子的 DataFrame
            factor_cols: 需要正交化的因子列表
            method: 正交化方法（'pca' 或 'gram_schmidt'）
            n_components: PCA 保留的主成分数量（None 表示保留全部）

        Returns:
            正交化后的 DataFrame（原始列 + _orth 后缀列）
        """
        result = df.copy()

        # 提取因子矩阵
        valid_mask = result[factor_cols].notna().all(axis=1)
        if valid_mask.sum() < 10:
            print("警告: 有效样本不足，跳过因子正交化")
            return result

        X = result.loc[valid_mask, factor_cols].values
        self.feature_names = factor_cols

        if method == "pca":
            # PCA 正交化
            n_comp = n_components or len(factor_cols)
            pca = PCA(n_components=n_comp)
            X_orth = pca.fit_transform(X)

            # 保存正交化矩阵
            self.orthogonalization_matrix = pca.components_.T

            # 保存正交化后的因子
            for i in range(X_orth.shape[1]):
                result.loc[valid_mask, f"factor_pc{i+1}"] = X_orth[:, i]

        elif method == "gram_schmidt":
            # Gram-Schmidt 正交化
            X_orth = self._gram_schmidt_orthogonalization(X)

            # 保存正交化后的因子
            for i, factor in enumerate(factor_cols):
                result.loc[valid_mask, f"{factor}_orth"] = X_orth[:, i]

        else:
            raise ValueError(f"不支持的正交化方法: {method}")

        return result

    def _gram_schmidt_orthogonalization(self, X: np.ndarray) -> np.ndarray:
        """Gram-Schmidt 正交化

        Args:
            X: 输入矩阵 (n_samples, n_features)

        Returns:
            正交化后的矩阵
        """
        n_samples, n_features = X.shape
        Q = np.zeros_like(X)

        for i in range(n_features):
            # 取第 i 列
            q = X[:, i].copy()

            # 减去前面所有正交向量的投影
            for j in range(i):
                q = q - np.dot(q, Q[:, j]) * Q[:, j]

            # 归一化
            norm = np.linalg.norm(q)
            if norm > 1e-10:
                Q[:, i] = q / norm
            else:
                Q[:, i] = q

        return Q

    # ==================== 多 Horizon 标签融合 ====================

    def multi_horizon_label_fusion(
        self,
        df: pd.DataFrame,
        label_cols: List[str],
        weights: Optional[List[float]] = None,
        risk_adjusted: bool = True,
    ) -> pd.DataFrame:
        """多 Horizon 标签融合

        Args:
            df: 包含多个标签的 DataFrame
            label_cols: 标签列名列表（如 ['label_5d', 'label_10d', 'label_20d']）
            weights: 权重列表（如 [0.2, 0.3, 0.5]），None 表示等权
            risk_adjusted: 是否进行风险调整（除以波动率）

        Returns:
            添加融合标签的 DataFrame
        """
        result = df.copy()

        # 检查标签列是否存在
        missing_cols = [col for col in label_cols if col not in df.columns]
        if missing_cols:
            print(f"警告: 缺少标签列 {missing_cols}，跳过标签融合")
            return result

        # 默认等权
        if weights is None:
            weights = [1.0 / len(label_cols)] * len(label_cols)

        if len(weights) != len(label_cols):
            raise ValueError("权重数量必须与标签数量一致")

        # 风险调整
        if risk_adjusted:
            adjusted_labels = []
            for col in label_cols:
                std = result[col].std()
                if std > 0:
                    adjusted_labels.append(result[col] / std)
                else:
                    adjusted_labels.append(result[col])
        else:
            adjusted_labels = [result[col] for col in label_cols]

        # 加权融合
        fused_label = sum(w * label for w, label in zip(weights, adjusted_labels))
        result["label_fused"] = fused_label

        return result

    # ==================== 行业中性化（已有，补充完整） ====================

    def industry_neutralization(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        industry_col: str = "industry",
    ) -> pd.DataFrame:
        """行业中性化处理

        对每个因子减去行业均值

        Args:
            df: 包含因子和行业的 DataFrame
            factor_cols: 需要中性化的因子列表
            industry_col: 行业列名

        Returns:
            中性化后的 DataFrame（原始列 + _ind_neutral 后缀列）
        """
        result = df.copy()

        if industry_col not in df.columns:
            print(f"警告: 缺少行业列 {industry_col}，跳过行业中性化")
            return result

        for factor in factor_cols:
            if factor not in df.columns:
                continue

            # 计算行业均值
            industry_mean = result.groupby(industry_col)[factor].transform("mean")

            # 减去行业均值
            result[f"{factor}_ind_neutral"] = result[factor] - industry_mean

        return result

    def industry_neutralize(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        industry_col: str = "industry",
    ) -> pd.DataFrame:
        """兼容旧接口：行业中性化"""
        return self.industry_neutralization(
            df=df,
            factor_cols=factor_cols,
            industry_col=industry_col,
        )

    # ==================== 双重中性化（行业 + 市值） ====================

    def double_neutralization(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        industry_col: str = "industry",
        market_cap_col: str = "total_mv",
    ) -> pd.DataFrame:
        """双重中性化（行业 + 市值）

        先做行业中性化，再做市值中性化

        Args:
            df: 包含因子、行业和市值的 DataFrame
            factor_cols: 需要中性化的因子列表
            industry_col: 行业列名
            market_cap_col: 市值列名

        Returns:
            中性化后的 DataFrame
        """
        # 1. 行业中性化
        result = self.industry_neutralization(df, factor_cols, industry_col)

        # 2. 市值中性化（对行业中性化后的因子）
        ind_neutral_cols = [f"{col}_ind_neutral" for col in factor_cols if f"{col}_ind_neutral" in result.columns]
        if ind_neutral_cols:
            result = self.market_cap_neutralization(result, ind_neutral_cols, market_cap_col)

            # 重命名为双重中性化
            for col in ind_neutral_cols:
                if f"{col}_neutral" in result.columns:
                    result = result.rename(
                        columns={f"{col}_neutral": f"{col.replace('_ind_neutral', '')}_double_neutral"}
                    )

        return result
