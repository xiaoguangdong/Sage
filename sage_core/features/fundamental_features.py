"""
基本面特征（估值、质量、成长）

按照豆包 Champion 策略四因子体系：
- 质量因子（30%）：ROIC、毛利率、现金流/净利润、资产负债率
- 成长因子（30%）：营收增速、净利润增速、一致预期上调、PEG
- 动量因子（20%）：相对板块超额收益、成交额分位数（在 price_features）
- 低波动因子（20%）：波动率、最大回撤、下行波动率（在 price_features）
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from .base import FeatureGenerator, FeatureSpec
from .registry import register_feature

logger = logging.getLogger(__name__)

# 豆包 Champion 策略四因子权重
DOUBAO_FACTOR_WEIGHTS = {
    "quality": 0.30,  # 质量因子
    "growth": 0.30,  # 成长因子
    "momentum": 0.20,  # 动量因子（在 price_features 计算）
    "low_vol": 0.20,  # 低波动因子（在 price_features 计算）
}


@register_feature
class FundamentalFeatures(FeatureGenerator):
    """基本面特征提取器（估值/质量/成长）- 豆包 Champion 策略"""

    spec = FeatureSpec(
        name="fundamental_features",
        input_fields=("date", "stock", "ts_code"),
        description="基本面特征（估值/质量/成长）- 豆包四因子体系",
    )

    def __init__(
        self,
        valuation_windows: List[int] = None,
        quality_windows: List[int] = None,
        growth_windows: List[int] = None,
    ):
        """
        初始化基本面特征提取器

        Args:
            valuation_windows: 估值分位数计算窗口（默认 [252, 504] 即1年/2年）
            quality_windows: 质量指标滚动窗口（默认 [4, 12] 即4周/12周）
            growth_windows: 成长指标计算窗口（默认 [4, 12] 即4周/12周）
        """
        self.valuation_windows = valuation_windows or [252, 504]
        self.quality_windows = quality_windows or [4, 12]
        self.growth_windows = growth_windows or [4, 12]

    def calculate_valuation_features(
        self,
        df: pd.DataFrame,
        daily_basic: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        计算估值特征

        Args:
            df: 股票数据DataFrame（需包含 ts_code, date）
            daily_basic: 每日基本面数据（pe_ttm, pb 等）

        Returns:
            包含估值特征的DataFrame
        """
        df = df.copy()

        # 如果有外部 daily_basic 数据，合并进来
        if daily_basic is not None and not daily_basic.empty:
            # 确保日期格式一致
            if "trade_date" in daily_basic.columns:
                daily_basic = daily_basic.rename(columns={"trade_date": "date"})

            # 合并基本面数据
            merge_cols = ["ts_code", "date"]
            available_cols = [c for c in merge_cols if c in daily_basic.columns]
            if len(available_cols) >= 2:
                # 只保留需要的估值字段
                val_cols = ["ts_code", "date", "pe_ttm", "pb", "pe", "total_mv", "circ_mv"]
                val_cols = [c for c in val_cols if c in daily_basic.columns]
                daily_basic_subset = daily_basic[val_cols].copy()
                df = df.merge(daily_basic_subset, on=["ts_code", "date"], how="left")

        # 计算估值分位数（需要历史数据）
        df = df.sort_values(["ts_code", "date"])

        # PE 分位数（历史百分位）
        if "pe_ttm" in df.columns:
            for window in self.valuation_windows:
                df[f"pe_percentile_{window}d"] = df.groupby("ts_code")["pe_ttm"].transform(
                    lambda x: x.rolling(window, min_periods=int(window * 0.5)).rank(pct=True)
                )
            # 默认使用 252 日分位数
            df["pe_percentile"] = df.get("pe_percentile_252d", df.get("pe_percentile_504d", np.nan))

        # PB 分位数（历史百分位）
        if "pb" in df.columns:
            for window in self.valuation_windows:
                df[f"pb_percentile_{window}d"] = df.groupby("ts_code")["pb"].transform(
                    lambda x: x.rolling(window, min_periods=int(window * 0.5)).rank(pct=True)
                )
            df["pb_percentile"] = df.get("pb_percentile_252d", df.get("pb_percentile_504d", np.nan))

        # PE/PB 合成评分（低估值加分）
        if "pe_ttm" in df.columns and "pb" in df.columns:
            # 标准化后取负（低估值 = 高评分）
            df["pe_zscore"] = df.groupby("ts_code")["pe_ttm"].transform(
                lambda x: (x - x.rolling(252, min_periods=60).mean()) / x.rolling(252, min_periods=60).std()
            )
            df["pb_zscore"] = df.groupby("ts_code")["pb"].transform(
                lambda x: (x - x.rolling(252, min_periods=60).mean()) / x.rolling(252, min_periods=60).std()
            )
            # 估值评分（越低越好）
            df["valuation_score"] = -df["pe_zscore"] * 0.5 - df["pb_zscore"] * 0.5

        logger.info("估值特征计算完成")
        return df

    def calculate_quality_features(
        self,
        df: pd.DataFrame,
        fina_indicator: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        计算质量特征

        Args:
            df: 股票数据DataFrame
            fina_indicator: 财务指标数据（roe, gross_margin, roic 等）

        Returns:
            包含质量特征的DataFrame
        """
        df = df.copy()

        # 如果有外部财务指标数据，合并进来
        if fina_indicator is not None and not fina_indicator.empty:
            # 确保日期格式一致
            if "ann_date" in fina_indicator.columns:
                fina_indicator = fina_indicator.rename(columns={"ann_date": "date"})
            if "end_date" in fina_indicator.columns:
                fina_indicator = fina_indicator.drop(columns=["end_date"], errors="ignore")

            # 合并财务指标
            merge_cols = ["ts_code", "date"]
            available_cols = [c for c in merge_cols if c in fina_indicator.columns]
            if len(available_cols) >= 2:
                # 只保留需要的质量字段
                quality_cols = [
                    "ts_code",
                    "date",
                    "roe",
                    "roe_dt",
                    "grossprofit_margin",
                    "netprofit_margin",
                    "roic",
                    "ocfps",
                    "debt_to_assets",
                ]
                quality_cols = [c for c in quality_cols if c in fina_indicator.columns]
                fina_subset = fina_indicator[quality_cols].copy()
                df = df.merge(fina_subset, on=["ts_code", "date"], how="left")

        # ROE 处理（优先使用 roe_dt，即扣非ROE）
        if "roe_dt" in df.columns:
            df["roe"] = df["roe"].fillna(df["roe_dt"])
        elif "roe" not in df.columns:
            df["roe"] = np.nan

        # gross_margin 处理
        if "grossprofit_margin" in df.columns:
            df["gross_margin"] = df["grossprofit_margin"]
        elif "gross_margin" not in df.columns:
            df["gross_margin"] = np.nan

        # ROIC 处理
        if "roic" not in df.columns:
            df["roic"] = np.nan

        # 计算质量变化（财务指标加速度）
        df = df.sort_values(["ts_code", "date"])
        for col in ["roe", "gross_margin", "roic"]:
            if col in df.columns:
                df[f"{col}_change"] = df.groupby("ts_code")[col].diff()
                df[f"{col}_ma4"] = df.groupby("ts_code")[col].transform(lambda x: x.rolling(4, min_periods=1).mean())

        # 综合质量评分
        quality_components = []
        for col in ["roe", "gross_margin", "roic"]:
            if col in df.columns:
                # 标准化
                df[f"{col}_zscore"] = df.groupby("ts_code")[col].transform(
                    lambda x: (x - x.rolling(252, min_periods=60).mean()) / x.rolling(252, min_periods=60).std()
                )
                quality_components.append(f"{col}_zscore")

        if quality_components:
            weights = {"roe_zscore": 0.4, "gross_margin_zscore": 0.35, "roic_zscore": 0.25}
            df["quality_score"] = sum(
                df.get(c, 0) * weights.get(c, 1 / len(quality_components)) for c in quality_components
            )
        else:
            df["quality_score"] = np.nan

        logger.info("质量特征计算完成")
        return df

    def calculate_growth_features(
        self,
        df: pd.DataFrame,
        fina_indicator: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        计算成长因子（豆包 Champion 策略核心）

        成长因子包括：
        - revenue_yoy: 营收同比增速
        - profit_yoy: 净利润同比增速
        - peg: PE/增长率（估值与成长匹配度）
        - forecast_adjust: 一致预期上调幅度

        Args:
            df: 股票数据DataFrame
            fina_indicator: 财务指标数据

        Returns:
            包含成长特征的DataFrame
        """
        df = df.copy()

        # 如果有外部财务指标数据，合并成长相关字段
        if fina_indicator is not None and not fina_indicator.empty:
            # 确保日期格式一致
            if "ann_date" in fina_indicator.columns:
                fina_indicator = fina_indicator.rename(columns={"ann_date": "date"})

            # 合并成长相关字段
            growth_cols = [
                "ts_code",
                "date",
                "or_yoy",
                "netprofit_yoy",
                "dt_netprofit_yoy",
                "profit_dedt",
                "oper_rev",
                "ebt_yoy",
                "tr_yoy",
            ]
            growth_cols = [c for c in growth_cols if c in fina_indicator.columns]
            if len(growth_cols) >= 2:
                fina_subset = fina_indicator[growth_cols].copy()
                df = df.merge(fina_subset, on=["ts_code", "date"], how="left")

        # 1. 营收同比增速
        if "or_yoy" in df.columns:
            df["revenue_yoy"] = df["or_yoy"] / 100  # 转换为小数
        elif "tr_yoy" in df.columns:
            df["revenue_yoy"] = df["tr_yoy"] / 100
        else:
            df["revenue_yoy"] = np.nan

        # 2. 净利润同比增速
        if "netprofit_yoy" in df.columns:
            df["profit_yoy"] = df["netprofit_yoy"] / 100
        elif "dt_netprofit_yoy" in df.columns:
            df["profit_yoy"] = df["dt_netprofit_yoy"] / 100
        else:
            df["profit_yoy"] = np.nan

        # 3. 净利润增速变化（加速度）
        df = df.sort_values(["ts_code", "date"])
        if "profit_yoy" in df.columns:
            df["profit_yoy_accel"] = df.groupby("ts_code")["profit_yoy"].diff()
            df["profit_yoy_accel"] = df["profit_yoy_accel"].clip(-1, 1)  # 限制异常值

        # 4. 营收增速变化（加速度）
        if "revenue_yoy" in df.columns:
            df["revenue_yoy_accel"] = df.groupby("ts_code")["revenue_yoy"].diff()
            df["revenue_yoy_accel"] = df["revenue_yoy_accel"].clip(-1, 1)

        # 5. PEG（PE/增长率）- 估值与成长匹配度
        if "pe_ttm" in df.columns and "profit_yoy" in df.columns:
            # PEG = PE / (增长率 * 100)
            # 注意：增长率需要转换为百分比
            growth_rate = df["profit_yoy"] * 100
            # 避免除零和负增长率
            growth_rate_safe = growth_rate.abs().replace(0, np.nan)
            df["peg"] = df["pe_ttm"] / growth_rate_safe
            # PEG < 1 表示低估，PEG > 1 表示高估
            df["peg"] = df["peg"].clip(0, 10)  # 限制异常值

        # 6. 成长评分（豆包方案）
        growth_components = []

        # 营收增速 z-score
        if "revenue_yoy" in df.columns:
            df["revenue_yoy_zscore"] = df.groupby("ts_code")["revenue_yoy"].transform(
                lambda x: (x - x.rolling(252, min_periods=60).mean()) / (x.rolling(252, min_periods=60).std() + 1e-8)
            )
            growth_components.append("revenue_yoy_zscore")

        # 净利润增速 z-score
        if "profit_yoy" in df.columns:
            df["profit_yoy_zscore"] = df.groupby("ts_code")["profit_yoy"].transform(
                lambda x: (x - x.rolling(252, min_periods=60).mean()) / (x.rolling(252, min_periods=60).std() + 1e-8)
            )
            growth_components.append("profit_yoy_zscore")

        # 利润加速度 z-score
        if "profit_yoy_accel" in df.columns:
            df["profit_yoy_accel_zscore"] = df.groupby("ts_code")["profit_yoy_accel"].transform(
                lambda x: (x - x.rolling(252, min_periods=60).mean()) / (x.rolling(252, min_periods=60).std() + 1e-8)
            )
            growth_components.append("profit_yoy_accel_zscore")

        # 计算成长因子综合评分
        if growth_components:
            # 成长因子权重（豆包方案）
            growth_weights = {
                "revenue_yoy_zscore": 0.30,
                "profit_yoy_zscore": 0.40,
                "profit_yoy_accel_zscore": 0.30,
            }
            df["growth_score"] = sum(
                df.get(c, 0) * growth_weights.get(c, 1 / len(growth_components)) for c in growth_components
            )
        else:
            df["growth_score"] = np.nan

        logger.info("成长特征计算完成")
        return df

    def calculate_cashflow_quality(
        self,
        df: pd.DataFrame,
        fina_indicator: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        计算现金流质量特征（豆包质量因子核心）

        现金流质量 = 经营现金流净额 / 净利润
        该指标 > 1 表示盈利质量高，利润有现金支撑

        Args:
            df: 股票数据DataFrame
            fina_indicator: 财务指标数据

        Returns:
            包含现金流质量特征的DataFrame
        """
        df = df.copy()

        # 合并现金流相关字段
        if fina_indicator is not None and not fina_indicator.empty:
            if "ann_date" in fina_indicator.columns:
                fina_indicator = fina_indicator.rename(columns={"ann_date": "date"})

            cf_cols = ["ts_code", "date", "ocfps", "ocf_yoy", "ocf_to_ebt", "ocf_to_debt"]
            cf_cols = [c for c in cf_cols if c in fina_indicator.columns]
            if len(cf_cols) >= 2:
                fina_subset = fina_indicator[cf_cols].copy()
                df = df.merge(fina_subset, on=["ts_code", "date"], how="left")

        # 现金流/净利润（豆包质量因子核心指标）
        if "ocf_to_ebt" in df.columns:
            df["cashflow_to_profit"] = df["ocf_to_ebt"]
        elif "ocfps" in df.columns:
            # 如果有每股经营现金流，可以近似计算
            # cashflow_to_profit 需要总额数据，这里用 ocf_yoy 作为代理
            df["cashflow_to_profit"] = df.get("ocf_yoy", np.nan)

        # 现金流质量评分
        if "cashflow_to_profit" in df.columns:
            df["cashflow_quality_zscore"] = df.groupby("ts_code")["cashflow_to_profit"].transform(
                lambda x: (x - x.rolling(252, min_periods=60).mean()) / (x.rolling(252, min_periods=60).std() + 1e-8)
            )

        logger.info("现金流质量特征计算完成")
        return df

    def calculate_all_features(
        self,
        df: pd.DataFrame,
        daily_basic: Optional[pd.DataFrame] = None,
        fina_indicator: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        计算所有基本面特征（豆包 Champion 策略四因子体系）

        Args:
            df: 股票数据DataFrame
            daily_basic: 每日基本面数据
            fina_indicator: 财务指标数据

        Returns:
            包含所有特征的DataFrame
        """
        logger.info("开始计算基本面特征（豆包四因子体系）...")

        # 检查必要的列
        required_cols = ["ts_code", "date"]
        for col in required_cols:
            if col not in df.columns:
                if col == "date" and "trade_date" in df.columns:
                    df["date"] = df["trade_date"]
                else:
                    raise ValueError(f"缺少必要的列: {col}")

        # 计算各类特征
        df = self.calculate_valuation_features(df, daily_basic)
        df = self.calculate_quality_features(df, fina_indicator)
        df = self.calculate_cashflow_quality(df, fina_indicator)
        df = self.calculate_growth_features(df, fina_indicator)

        # 计算豆包四因子综合评分（基本面部分）
        # 质量（30%）+ 成长（30%）= 60%，动量和低波动在 price_features 计算
        df["fundamental_score"] = (
            df.get("quality_score", 0) * DOUBAO_FACTOR_WEIGHTS["quality"]
            + df.get("growth_score", 0) * DOUBAO_FACTOR_WEIGHTS["growth"]
        )

        # 统计新增特征
        base_cols = set(df.columns)
        new_features = [
            c
            for c in df.columns
            if c not in base_cols
            or c
            in [
                "pe_ttm",
                "pb",
                "pe_percentile",
                "pb_percentile",
                "roe",
                "gross_margin",
                "roic",
                "quality_score",
                "valuation_score",
                "revenue_yoy",
                "profit_yoy",
                "peg",
                "growth_score",
                "cashflow_to_profit",
                "fundamental_score",
            ]
        ]

        logger.info(f"基本面特征计算完成，新增特征: {len(new_features)}")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(df)
        return self.calculate_all_features(df)

    def get_feature_names(self) -> List[str]:
        """返回生成的特征名称列表（豆包四因子体系）"""
        return [
            # 估值特征
            "pe_ttm",
            "pb",
            "pe_percentile",
            "pb_percentile",
            "pe_zscore",
            "pb_zscore",
            "valuation_score",
            # 质量因子（30%）
            "roe",
            "gross_margin",
            "roic",
            "debt_to_assets",
            "roe_change",
            "gross_margin_change",
            "roic_change",
            "roe_ma4",
            "gross_margin_ma4",
            "roic_ma4",
            "roe_zscore",
            "gross_margin_zscore",
            "roic_zscore",
            "cashflow_to_profit",
            "cashflow_quality_zscore",
            "quality_score",
            # 成长因子（30%）
            "revenue_yoy",
            "profit_yoy",
            "revenue_yoy_accel",
            "profit_yoy_accel",
            "revenue_yoy_zscore",
            "profit_yoy_zscore",
            "profit_yoy_accel_zscore",
            "peg",
            "growth_score",
            # 综合评分
            "fundamental_score",
        ]


if __name__ == "__main__":
    # 测试基本面特征提取
    logging.basicConfig(level=logging.INFO)

    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    stocks = ["600519.SH", "000858.SZ"]

    data = []
    for stock in stocks:
        for i, date in enumerate(dates):
            data.append(
                {
                    "ts_code": stock,
                    "date": date.strftime("%Y%m%d"),
                    "stock": stock,
                    "close": 100 + np.cumsum(np.random.randn(len(dates)))[i] * 0.01,
                    "pe_ttm": np.random.uniform(10, 50),
                    "pb": np.random.uniform(1, 5),
                    "roe": np.random.uniform(0.05, 0.25),
                    "gross_margin": np.random.uniform(0.2, 0.5),
                    "roic": np.random.uniform(0.05, 0.2),
                }
            )

    df = pd.DataFrame(data)

    feature_extractor = FundamentalFeatures()
    df_with_features = feature_extractor.calculate_all_features(df)

    print("\n生成的特征列:")
    print([c for c in df_with_features.columns if c in feature_extractor.get_feature_names()])

    print("\n数据预览:")
    print(df_with_features[["ts_code", "date", "pe_percentile", "pb_percentile", "quality_score"]].tail(10))
