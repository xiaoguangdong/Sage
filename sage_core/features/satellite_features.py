"""
卫星策略特征模块

按照卫星股相关讨论文档实现四因子评分体系：
- 景气成长因子（40%）：业绩增速、盈利能力、行业景气度
- 事件驱动因子（30%）：核心事件、预期差、事件确定性
- 量价动量因子（20%）：趋势强度、动量效应、量价配合
- 筹码结构因子（10%）：筹码集中度、资金认可度

卫星仓位管理：
- 总仓位上限：20%-30%
- 景气成长型：初始≤卫星总仓20%，上限30%
- 事件驱动型：初始≤卫星总仓10%，上限15%
- ST类：单只≤5%，合计≤10%
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sage_core.utils.logging_utils import format_task_summary, setup_logging

from .base import FeatureGenerator, FeatureSpec
from .registry import register_feature

logger = logging.getLogger(__name__)

# 卫星策略四因子权重（100分制）
SATELLITE_FACTOR_WEIGHTS = {
    "growth": 40,  # 景气成长因子
    "event": 30,  # 事件驱动因子
    "momentum": 20,  # 量价动量因子
    "chip": 10,  # 筹码结构因子
}

# 卫星仓位管理配置
SATELLITE_POSITION_LIMITS = {
    "total_max_ratio": 0.30,  # 卫星仓位总上限（总资金30%）
    "growth_init_ratio": 0.20,  # 景气成长型初始仓位（卫星总仓20%）
    "growth_max_ratio": 0.30,  # 景气成长型上限（卫星总仓30%）
    "event_init_ratio": 0.10,  # 事件驱动型初始仓位（卫星总仓10%）
    "event_max_ratio": 0.15,  # 事件驱动型上限（卫星总仓15%）
    "st_single_ratio": 0.05,  # ST单只上限（卫星总仓5%）
    "st_total_ratio": 0.10,  # ST合计上限（卫星总仓10%）
}


@register_feature
class SatelliteFeatures(FeatureGenerator):
    """卫星策略特征提取器（四因子评分体系）"""

    spec = FeatureSpec(
        name="satellite_features",
        input_fields=("date", "stock", "ts_code", "close"),
        description="卫星策略特征（景气成长/事件驱动/量价动量/筹码结构）",
    )

    def __init__(
        self,
        lookback_days: int = 120,
        min_score: int = 60,
    ):
        """
        初始化卫星策略特征提取器

        Args:
            lookback_days: 回看天数（默认120天）
            min_score: 精选池最低评分（默认60分）
        """
        self.lookback_days = lookback_days
        self.min_score = min_score

    def calculate_growth_factor(
        self,
        df: pd.DataFrame,
        fina_indicator: Optional[pd.DataFrame] = None,
        industry_rank: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        计算景气成长因子（40分）

        细分因子：
        - 业绩增速（25分）：近2个季度净利润同比≥30%、营收同比≥20%
        - 盈利能力（10分）：毛利率≥行业均值、净利率≥10%、ROE≥8%
        - 行业景气度（5分）：所属行业处于景气度前30%

        Args:
            df: 股票数据DataFrame
            fina_indicator: 财务指标数据
            industry_rank: 行业景气度排名数据

        Returns:
            包含景气成长因子评分的DataFrame
        """
        df = df.copy()
        df["growth_factor_score"] = 0.0

        # 合并财务指标数据
        if fina_indicator is not None and not fina_indicator.empty:
            if "ann_date" in fina_indicator.columns:
                fina_indicator = fina_indicator.rename(columns={"ann_date": "date"})

            # 取每只股票最近2期财报
            fina_cols = [
                "ts_code",
                "date",
                "tr_yoy",
                "or_yoy",
                "netprofit_yoy",
                "roe_ttm",
                "grossprofit_margin",
                "netprofit_margin",
                "net_cash_operate_act",
                "net_profit",
            ]
            fina_cols = [c for c in fina_cols if c in fina_indicator.columns]
            if len(fina_cols) >= 3:
                fina_subset = fina_indicator[fina_cols].copy()
                # 取最近2期
                fina_subset = fina_subset.sort_values(["ts_code", "date"])
                fina_latest_2q = fina_subset.groupby("ts_code").tail(2)

                # 合并到主数据
                df = df.merge(fina_latest_2q, on=["ts_code", "date"], how="left", suffixes=("", "_fina"))

        # 1. 业绩增速评分（25分）
        def calc_growth_score_25(row) -> float:
            score = 0.0
            # 净利润同比增速
            profit_yoy = row.get("netprofit_yoy", np.nan)
            # 营收同比增速
            revenue_yoy = row.get("or_yoy", row.get("tr_yoy", np.nan))

            if pd.notna(profit_yoy) and pd.notna(revenue_yoy):
                # 近2季度平均增速达标
                if profit_yoy >= 30 and revenue_yoy >= 20:
                    score += 15
                    # 增速环比提升（需要2期数据，简化处理）
                    if profit_yoy > 30:
                        score += min((profit_yoy - 30) / 10, 5)  # 每超10%加2分，最高5分

            # 现金流匹配度
            ocf = row.get("net_cash_operate_act", np.nan)
            net_profit = row.get("net_profit", np.nan)
            if pd.notna(ocf) and pd.notna(net_profit) and net_profit > 0:
                ocf_ratio = ocf / net_profit
                if ocf_ratio >= 0.8:
                    score += 5

            return min(score, 25)

        df["profit_growth_score"] = df.apply(calc_growth_score_25, axis=1)

        # 2. 盈利能力评分（10分）
        def calc_profitability_score(row) -> float:
            score = 0.0
            # 毛利率≥行业均值（简化为≥20%）
            gross_margin = row.get("grossprofit_margin", np.nan)
            if pd.notna(gross_margin) and gross_margin >= 20:
                score += 3

            # 净利率≥10%
            net_margin = row.get("netprofit_margin", np.nan)
            if pd.notna(net_margin) and net_margin >= 10:
                score += 3

            # ROE(TTM)≥8%
            roe = row.get("roe_ttm", row.get("roe_dt", np.nan))
            if pd.notna(roe) and roe >= 8:
                score += 4

            return score

        df["profitability_score"] = df.apply(calc_profitability_score, axis=1)

        # 3. 行业景气度评分（5分）
        if industry_rank is not None and not industry_rank.empty:
            if "industry" in df.columns and "industry_name" in industry_rank.columns:
                df = df.merge(
                    industry_rank[["industry_name", "industry_rank"]],
                    left_on="industry",
                    right_on="industry_name",
                    how="left",
                )
                df["industry_score"] = df["industry_rank"].apply(lambda x: 5 if pd.notna(x) and x <= 0.3 else 0)
            else:
                df["industry_score"] = 0
        else:
            df["industry_score"] = 0

        # 景气成长因子总分
        df["growth_factor_score"] = df["profit_growth_score"] + df["profitability_score"] + df["industry_score"]

        logger.info(f"景气成长因子计算完成，平均分: {df['growth_factor_score'].mean():.2f}")
        return df

    def calculate_event_factor(
        self,
        df: pd.DataFrame,
        major_events: Optional[pd.DataFrame] = None,
        survey_data: Optional[pd.DataFrame] = None,
        rating_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        计算事件驱动因子（30分）

        细分因子：
        - 核心事件（15分）：并购重组、控制权变更、资产注入、重大订单
        - 预期差（10分）：机构评级上调、密集调研
        - 事件确定性（5分）：重组方案获监管受理等

        Args:
            df: 股票数据DataFrame
            major_events: 重大事件数据
            survey_data: 机构调研数据
            rating_data: 机构评级数据

        Returns:
            包含事件驱动因子评分的DataFrame
        """
        df = df.copy()
        df["event_factor_score"] = 0.0

        # 初始化事件相关列
        df["has_core_event"] = False
        df["has_rating_up"] = False
        df["has_survey"] = False
        df["event_certainty"] = False

        # 1. 核心事件评分（15分）
        if major_events is not None and not major_events.empty:
            # 核心利好事件类型
            core_event_types = ["重组", "并购", "资产注入", "控制权变更", "重大合同", "技术突破"]
            major_events["is_core"] = major_events.get("event_type", "").apply(
                lambda x: any(kw in str(x) for kw in core_event_types)
            )
            core_event_codes = major_events[major_events["is_core"]]["ts_code"].unique().tolist()
            df["has_core_event"] = df["ts_code"].isin(core_event_codes)

        # 2. 预期差评分（10分）
        if rating_data is not None and not rating_data.empty:
            # 机构评级上调/首次覆盖
            rating_up_codes = (
                rating_data[rating_data.get("rating_change", "").isin(["上调", "首次"])]["ts_code"].unique().tolist()
            )
            df["has_rating_up"] = df["ts_code"].isin(rating_up_codes)

        if survey_data is not None and not survey_data.empty:
            # 密集调研（近1个月调研机构≥10家）
            survey_codes = survey_data[survey_data.get("org_count", 0) >= 10]["ts_code"].unique().tolist()
            df["has_survey"] = df["ts_code"].isin(survey_codes)

        # 3. 事件确定性评分（5分）
        # 简化处理：有核心事件且有明确进展
        df["event_certainty"] = df["has_core_event"]

        # 计算事件驱动因子总分
        def calc_event_score(row) -> float:
            score = 0.0

            # 核心事件（15分）
            if row["has_core_event"]:
                score += 15

            # 预期差（10分）
            if row["has_rating_up"]:
                score += 6
            if row["has_survey"]:
                score += 4

            # 事件确定性（5分）
            if row["event_certainty"]:
                score += 5

            return min(score, 30)

        df["event_factor_score"] = df.apply(calc_event_score, axis=1)

        logger.info(f"事件驱动因子计算完成，平均分: {df['event_factor_score'].mean():.2f}")
        return df

    def calculate_momentum_factor(
        self,
        df: pd.DataFrame,
        index_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        计算量价动量因子（20分）

        细分因子：
        - 趋势强度（5分）：均线多头排列
        - 动量效应（8分）：近20日涨幅、超额收益
        - 量价配合（7分）：量能放大、价涨量增

        Args:
            df: 股票数据DataFrame
            index_data: 指数数据（用于计算超额收益）

        Returns:
            包含量价动量因子评分的DataFrame
        """
        df = df.copy()
        df = df.sort_values(["ts_code", "date"])

        # 计算均线
        df["ma20"] = df.groupby("ts_code")["close"].transform(lambda x: x.rolling(20, min_periods=10).mean())
        df["ma50"] = df.groupby("ts_code")["close"].transform(lambda x: x.rolling(50, min_periods=25).mean())

        # 计算成交量均值
        if "volume" in df.columns:
            df["vol_20d"] = df.groupby("ts_code")["volume"].transform(lambda x: x.rolling(20, min_periods=10).mean())
            df["vol_60d"] = df.groupby("ts_code")["volume"].transform(lambda x: x.rolling(60, min_periods=30).mean())

        # 计算收益率
        df["ret_20d"] = df.groupby("ts_code")["close"].transform(lambda x: x.pct_change(20) * 100)

        # 计算基准收益率
        if index_data is not None and not index_data.empty:
            index_ret_20d = (index_data["close"].iloc[-1] / index_data["close"].iloc[-20] - 1) * 100
        else:
            index_ret_20d = 0

        df["excess_ret_20d"] = df["ret_20d"] - index_ret_20d

        # 1. 趋势强度评分（5分）
        def calc_trend_score(row) -> float:
            close = row.get("close", np.nan)
            ma20 = row.get("ma20", np.nan)
            ma50 = row.get("ma50", np.nan)

            if pd.isna(close) or pd.isna(ma20):
                return 0

            # 多头排列：close > ma20 > ma50
            if pd.notna(ma50) and close > ma20 > ma50:
                return 5
            elif close > ma20:
                return 3
            return 0

        df["trend_score"] = df.apply(calc_trend_score, axis=1)

        # 2. 动量效应评分（8分）
        def calc_momentum_score(row) -> float:
            score = 0.0
            ret_20d = row.get("ret_20d", np.nan)
            excess_ret = row.get("excess_ret_20d", np.nan)

            # 近20日涨幅≥10%（全市场前30%近似）
            if pd.notna(ret_20d) and ret_20d >= 10:
                score += 4

            # 超额收益≥10%
            if pd.notna(excess_ret) and excess_ret >= 10:
                score += 4

            return min(score, 8)

        df["momentum_score"] = df.apply(calc_momentum_score, axis=1)

        # 3. 量价配合评分（7分）
        def calc_volume_score(row) -> float:
            score = 0.0
            vol_20d = row.get("vol_20d", np.nan)
            vol_60d = row.get("vol_60d", np.nan)
            ret_20d = row.get("ret_20d", np.nan)

            if pd.isna(vol_20d) or pd.isna(vol_60d) or vol_60d == 0:
                return 0

            vol_ratio = vol_20d / vol_60d

            # 量能放大≥50%
            if vol_ratio >= 1.5:
                score += 4
            elif vol_ratio >= 1:
                score += 2

            # 价涨量增（涨幅为正且量能放大）
            if pd.notna(ret_20d) and ret_20d > 0 and vol_ratio >= 1:
                score += 3

            return min(score, 7)

        df["volume_score"] = df.apply(calc_volume_score, axis=1)

        # 量价动量因子总分
        df["momentum_factor_score"] = df["trend_score"] + df["momentum_score"] + df["volume_score"]

        logger.info(f"量价动量因子计算完成，平均分: {df['momentum_factor_score'].mean():.2f}")
        return df

    def calculate_chip_factor(
        self,
        df: pd.DataFrame,
        holder_data: Optional[pd.DataFrame] = None,
        top_list_data: Optional[pd.DataFrame] = None,
        northbound_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        计算筹码结构因子（10分）

        细分因子：
        - 筹码集中度（5分）：股东人数持续减少
        - 资金认可度（5分）：机构净买入、北向增持

        Args:
            df: 股票数据DataFrame
            holder_data: 股东人数数据
            top_list_data: 龙虎榜数据
            northbound_data: 北向资金数据

        Returns:
            包含筹码结构因子评分的DataFrame
        """
        df = df.copy()

        # 初始化筹码相关列
        df["holder_decreasing"] = False
        df["org_net_buy"] = False
        df["northbound_increase"] = False

        # 1. 筹码集中度评分（5分）
        if holder_data is not None and not holder_data.empty:
            # 取近3个季度股东人数数据
            holder_data = holder_data.sort_values(["ts_code", "end_date"])
            holder_latest_3q = holder_data.groupby("ts_code").tail(3)

            # 判断是否连续减少
            def is_holder_decreasing(group):
                if len(group) < 2:
                    return False
                holder_nums = group["holder_num"].tolist()
                # 连续2个季度减少
                return holder_nums[-1] < holder_nums[-2]

            decreasing_codes = holder_latest_3q.groupby("ts_code").apply(is_holder_decreasing)
            decreasing_codes = decreasing_codes[decreasing_codes].index.tolist()
            df["holder_decreasing"] = df["ts_code"].isin(decreasing_codes)

        # 2. 资金认可度评分（5分）
        if top_list_data is not None and not top_list_data.empty:
            # 机构专用席位净买入为正
            org_buy = (
                top_list_data[top_list_data.get("exalter", "").str.contains("机构专用", na=False)]
                .groupby("ts_code")["net_buy"]
                .sum()
            )
            org_buy_codes = org_buy[org_buy > 0].index.tolist()
            df["org_net_buy"] = df["ts_code"].isin(org_buy_codes)

        if northbound_data is not None and not northbound_data.empty:
            # 北向资金持续增持
            northbound_increase_codes = (
                northbound_data[northbound_data.get("hold_ratio_change", 0) > 0]["ts_code"].unique().tolist()
            )
            df["northbound_increase"] = df["ts_code"].isin(northbound_increase_codes)

        # 计算筹码结构因子总分
        def calc_chip_score(row) -> float:
            score = 0.0

            # 筹码集中度（5分）
            if row["holder_decreasing"]:
                score += 5

            # 资金认可度（5分）
            if row["org_net_buy"] or row["northbound_increase"]:
                score += 5

            return min(score, 10)

        df["chip_factor_score"] = df.apply(calc_chip_score, axis=1)

        logger.info(f"筹码结构因子计算完成，平均分: {df['chip_factor_score'].mean():.2f}")
        return df

    def calculate_satellite_score(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        计算卫星策略综合评分（100分制）

        四因子权重：
        - 景气成长因子：40分
        - 事件驱动因子：30分
        - 量价动量因子：20分
        - 筹码结构因子：10分

        Args:
            df: 包含各因子评分的DataFrame

        Returns:
            包含综合评分的DataFrame
        """
        df = df.copy()

        # 确保各因子评分存在
        for col in ["growth_factor_score", "event_factor_score", "momentum_factor_score", "chip_factor_score"]:
            if col not in df.columns:
                df[col] = 0

        # 计算综合评分
        df["satellite_score"] = (
            df["growth_factor_score"] + df["event_factor_score"] + df["momentum_factor_score"] + df["chip_factor_score"]
        )

        # 标记精选池标的（评分≥60）
        df["in_select_pool"] = df["satellite_score"] >= self.min_score

        # 标记标的类型
        def classify_stock(row) -> str:
            if row["event_factor_score"] >= 15:
                return "event_driven"  # 事件驱动型
            elif row["growth_factor_score"] >= 24:  # 40分的60%
                return "growth"  # 景气成长型
            else:
                return "mixed"  # 混合型

        df["stock_type"] = df.apply(classify_stock, axis=1)

        logger.info(f"卫星策略综合评分计算完成，平均分: {df['satellite_score'].mean():.2f}")
        logger.info(f"精选池标的数量: {df['in_select_pool'].sum()}/{len(df)}")

        return df

    def calculate_position_limits(
        self,
        df: pd.DataFrame,
        total_capital: float = 1000000,
    ) -> pd.DataFrame:
        """
        计算单只标的仓位限额

        Args:
            df: 包含评分和类型的DataFrame
            total_capital: 总资金

        Returns:
            包含仓位限额的DataFrame
        """
        df = df.copy()

        # 卫星总仓位上限
        satellite_total_limit = total_capital * SATELLITE_POSITION_LIMITS["total_max_ratio"]

        def calc_position_limit(row) -> Dict:
            stock_type = row.get("stock_type", "mixed")
            score = row.get("satellite_score", 0)

            # 根据类型确定初始仓位比例
            if stock_type == "growth":
                init_ratio = SATELLITE_POSITION_LIMITS["growth_init_ratio"]
                max_ratio = SATELLITE_POSITION_LIMITS["growth_max_ratio"]
            elif stock_type == "event_driven":
                init_ratio = SATELLITE_POSITION_LIMITS["event_init_ratio"]
                max_ratio = SATELLITE_POSITION_LIMITS["event_max_ratio"]
            else:
                init_ratio = 0.10
                max_ratio = 0.20

            # 按评分调整初始仓位
            score_factor = min(score / 100, 1.0)
            init_position = satellite_total_limit * init_ratio * score_factor
            max_position = satellite_total_limit * max_ratio

            return {
                "init_position": init_position,
                "max_position": max_position,
                "position_ratio": init_position / total_capital,
            }

        position_info = df.apply(calc_position_limit, axis=1, result_type="expand")
        df["init_position"] = position_info["init_position"]
        df["max_position"] = position_info["max_position"]
        df["position_ratio"] = position_info["position_ratio"]

        return df

    def get_feature_names(self) -> List[str]:
        """返回生成的特征名称列表"""
        return [
            # 景气成长因子（40分）
            "profit_growth_score",
            "profitability_score",
            "industry_score",
            "growth_factor_score",
            # 事件驱动因子（30分）
            "has_core_event",
            "has_rating_up",
            "has_survey",
            "event_certainty",
            "event_factor_score",
            # 量价动量因子（20分）
            "trend_score",
            "momentum_score",
            "volume_score",
            "momentum_factor_score",
            # 筹码结构因子（10分）
            "holder_decreasing",
            "org_net_buy",
            "northbound_increase",
            "chip_factor_score",
            # 综合评分
            "satellite_score",
            "in_select_pool",
            "stock_type",
            # 仓位限额
            "init_position",
            "max_position",
            "position_ratio",
        ]


if __name__ == "__main__":
    start_time = datetime.now().timestamp()
    failure_reason = None
    setup_logging("features")
    try:
        # 创建测试数据
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
        stocks = ["600519.SH", "000858.SZ", "300750.SZ"]

        data = []
        for stock in stocks:
            for i, date in enumerate(dates):
                data.append(
                    {
                        "ts_code": stock,
                        "date": date.strftime("%Y%m%d"),
                        "stock": stock,
                        "close": 100 + np.cumsum(np.random.randn(len(dates)))[i] * 0.01,
                        "volume": np.random.randint(1000000, 10000000),
                    }
                )

        df = pd.DataFrame(data)

        feature_extractor = SatelliteFeatures()
        df = feature_extractor.calculate_volume_features(df)
        df = feature_extractor.calculate_price_features(df)
        df = feature_extractor.calculate_momentum_factor(df)
        df = feature_extractor.calculate_satellite_score(df)

        print("\n生成的特征列:")
        print([c for c in df.columns if c in feature_extractor.get_feature_names()])

        print("\n最新数据预览:")
        latest = df[df["date"] == df["date"].max()]
        print(latest[["ts_code", "satellite_score", "stock_type", "in_select_pool"]].head())
    except Exception as exc:
        failure_reason = str(exc)
        raise
    finally:
        logger.info(
            format_task_summary(
                "satellite_features_demo",
                window=None,
                elapsed_s=datetime.now().timestamp() - start_time,
                error=failure_reason,
            )
        )
    df = feature_extractor.calculate_momentum_factor(df)
    df = feature_extractor.calculate_satellite_score(df)

    print("\n生成的特征列:")
    print([c for c in df.columns if c in feature_extractor.get_feature_names()])

    print("\n最新数据预览:")
    latest = df[df["date"] == df["date"].max()]
    print(latest[["ts_code", "satellite_score", "stock_type", "in_select_pool"]].head())
