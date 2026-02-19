#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
高 Alpha 因子的 FeatureGenerator 包装器

将 HighAlphaFeatures 适配到 FeatureGenerator 接口，使其能被 FeaturePipeline 调用。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from sage_core.features.base import FeatureGenerator, FeatureSpec
from sage_core.features.high_alpha_features import HighAlphaFeatures
from sage_core.features.registry import register_feature


@register_feature
class HighAlphaFeaturesWrapper(FeatureGenerator):
    """高 Alpha 因子包装器

    将 HighAlphaFeatures 适配到 FeatureGenerator 接口。
    支持选择性启用因子组，自动检测数据可用性。
    """

    spec = FeatureSpec(
        name="high_alpha_features",
        input_fields=("trade_date", "ts_code"),
        description="高Alpha因子：资金流细分、北向资金、融资融券、分析师预期",
    )

    def __init__(
        self,
        data_root: Path = Path("data/tushare"),
        enabled_groups: Optional[List[str]] = None,
        lookback_days: int = 20,
    ):
        """初始化

        Args:
            data_root: Tushare数据根目录
            enabled_groups: 启用的因子组列表，可选值：
                - "moneyflow": 资金流因子
                - "northbound": 北向资金因子
                - "margin": 融资融券因子
                - "analyst": 分析师预期因子
                默认 None 表示全部启用（但会检查数据可用性）
            lookback_days: 回溯天数，默认20天
        """
        self.data_root = Path(data_root)
        self.calculator = HighAlphaFeatures(self.data_root)
        self.lookback_days = lookback_days

        # 检查数据可用性
        available_groups = self._check_data_availability()

        # 确定最终启用的因子组
        if enabled_groups is None:
            self.enabled_groups = available_groups
        else:
            self.enabled_groups = [g for g in enabled_groups if g in available_groups]

        if not self.enabled_groups:
            print("⚠️  警告: 没有可用的高Alpha因子数据")
        else:
            print(f"✅ 启用高Alpha因子组: {', '.join(self.enabled_groups)}")

    def _check_data_availability(self) -> List[str]:
        """检查数据可用性

        Returns:
            可用的因子组列表
        """
        available = []

        # 检查 moneyflow
        moneyflow_dir = self.data_root / "moneyflow"
        if moneyflow_dir.exists() and list(moneyflow_dir.glob("*.parquet")):
            available.append("moneyflow")

        # 检查 northbound
        northbound_path = self.data_root / "northbound" / "northbound_hold.parquet"
        if northbound_path.exists():
            available.append("northbound")

        # 检查 margin
        margin_path = self.data_root / "margin.parquet"
        if margin_path.exists():
            available.append("margin")

        # 检查 analyst (forecast + express)
        forecast_dir = self.data_root / "fundamental" / "forecast"
        express_dir = self.data_root / "fundamental" / "express"
        if (forecast_dir.exists() and list(forecast_dir.glob("*.parquet"))) or (
            express_dir.exists() and list(express_dir.glob("*.parquet"))
        ):
            available.append("analyst")

        return available

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算高Alpha因子

        Args:
            df: 输入数据，必须包含 trade_date 和 ts_code 列

        Returns:
            添加了高Alpha因子列的DataFrame
        """
        if df.empty:
            return df

        # 验证输入
        self.validate_input(df)

        # 获取所有唯一的交易日期
        trade_dates = sorted(df["trade_date"].unique())

        all_features = []

        # 按日期计算因子
        for trade_date in trade_dates:
            date_features = {"trade_date": trade_date}

            # 1. 资金流因子
            if "moneyflow" in self.enabled_groups:
                try:
                    moneyflow_feat = self.calculator.compute_moneyflow_features(
                        trade_date=trade_date, lookback_days=self.lookback_days
                    )
                    if not moneyflow_feat.empty:
                        date_features["moneyflow"] = moneyflow_feat
                except Exception as e:
                    print(f"⚠️  计算资金流因子失败 ({trade_date}): {e}")

            # 2. 北向资金因子
            if "northbound" in self.enabled_groups:
                try:
                    northbound_feat = self.calculator.compute_northbound_features(
                        trade_date=trade_date, lookback_days=self.lookback_days * 3  # 北向数据用更长周期
                    )
                    if not northbound_feat.empty:
                        date_features["northbound"] = northbound_feat
                except Exception as e:
                    print(f"⚠️  计算北向资金因子失败 ({trade_date}): {e}")

            # 3. 融资融券因子
            if "margin" in self.enabled_groups:
                try:
                    margin_feat = self.calculator.compute_margin_features(
                        trade_date=trade_date, lookback_days=self.lookback_days
                    )
                    if not margin_feat.empty:
                        date_features["margin"] = margin_feat
                except Exception as e:
                    print(f"⚠️  计算融资融券因子失败 ({trade_date}): {e}")

            # 4. 分析师预期因子
            if "analyst" in self.enabled_groups:
                try:
                    analyst_feat = self.calculator.compute_analyst_features(
                        trade_date=trade_date, lookback_days=self.lookback_days * 6  # 分析师数据用更长周期
                    )
                    if not analyst_feat.empty:
                        date_features["analyst"] = analyst_feat
                except Exception as e:
                    print(f"⚠️  计算分析师预期因子失败 ({trade_date}): {e}")

            all_features.append(date_features)

        # 合并所有因子到原始DataFrame
        df = df.copy()

        for date_feat in all_features:
            trade_date = date_feat["trade_date"]
            mask = df["trade_date"] == trade_date

            # 合并各类因子
            for feat_type in ["moneyflow", "northbound", "margin", "analyst"]:
                if feat_type in date_feat:
                    feat_df = date_feat[feat_type]
                    # 按 ts_code 合并
                    for col in feat_df.columns:
                        if col != "ts_code":
                            df.loc[mask, col] = df.loc[mask, "ts_code"].map(feat_df.set_index("ts_code")[col])

        return df

    def fit(self, df: pd.DataFrame) -> "HighAlphaFeaturesWrapper":
        """拟合（高Alpha因子无需拟合）

        Args:
            df: 训练数据

        Returns:
            self
        """
        return self

    def get_feature_names(self) -> List[str]:
        """获取生成的特征名称列表

        Returns:
            特征名称列表
        """
        feature_names = []

        if "moneyflow" in self.enabled_groups:
            feature_names.extend(
                [
                    "large_net_inflow_ratio",
                    "main_inflow_days",
                    "retail_inst_divergence",
                    "super_large_ratio",
                ]
            )

        if "northbound" in self.enabled_groups:
            feature_names.extend(
                [
                    "config_fund_ratio",
                    "trading_fund_inflow",
                    "holding_concentration_change",
                    "holding_stability",
                ]
            )

        if "margin" in self.enabled_groups:
            feature_names.extend(
                [
                    "margin_balance_change_rate",
                    "margin_net_buy_ratio",
                ]
            )

        if "analyst" in self.enabled_groups:
            feature_names.extend(
                [
                    "analyst_upgrade_count",
                    "analyst_revision_magnitude",
                    "analyst_surprise_degree",
                    "analyst_consensus",
                ]
            )

        return feature_names
