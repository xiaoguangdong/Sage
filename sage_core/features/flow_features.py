"""
资金流向特征（北向资金、主力资金）
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from .base import FeatureGenerator, FeatureSpec
from .registry import register_feature

logger = logging.getLogger(__name__)


@register_feature
class FlowFeatures(FeatureGenerator):
    """资金流向特征提取器"""

    spec = FeatureSpec(
        name="flow_features",
        input_fields=("date", "stock", "ts_code"),
        description="资金流向特征（北向资金/主力资金/流动性）",
    )

    def __init__(
        self,
        northbound_windows: List[int] = None,
        flow_windows: List[int] = None,
    ):
        """
        初始化资金流向特征提取器

        Args:
            northbound_windows: 北向资金统计窗口（默认 [5, 20, 60]）
            flow_windows: 资金流向统计窗口（默认 [5, 20]）
        """
        self.northbound_windows = northbound_windows or [5, 20, 60]
        self.flow_windows = flow_windows or [5, 20]

    def calculate_northbound_features(
        self,
        df: pd.DataFrame,
        hk_hold: Optional[pd.DataFrame] = None,
        moneyflow_hsgt: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        计算北向资金特征

        Args:
            df: 股票数据DataFrame
            hk_hold: 港资持股数据（ts_code, trade_date, hold_amount, hold_ratio）
            moneyflow_hsgt: 北向资金净流入数据（trade_date, north_money 等）

        Returns:
            包含北向资金特征的DataFrame
        """
        df = df.copy()

        # 初始化北向资金字段
        df["northbound_hold_ratio"] = np.nan
        df["northbound_hold_amount"] = np.nan
        df["northbound_net_flow"] = np.nan

        # 合并港资持股数据（个股级）
        if hk_hold is not None and not hk_hold.empty:
            # 确保日期格式一致
            if "trade_date" in hk_hold.columns:
                hk_hold = hk_hold.rename(columns={"trade_date": "date"})

            # 合并持股比例
            hold_cols = ["ts_code", "date", "hold_ratio", "hold_amount"]
            hold_cols = [c for c in hold_cols if c in hk_hold.columns]
            if len(hold_cols) >= 3:
                hk_subset = hk_hold[hold_cols].copy()
                hk_subset = hk_subset.rename(
                    columns={"hold_ratio": "northbound_hold_ratio", "hold_amount": "northbound_hold_amount"}
                )
                df = df.merge(hk_subset, on=["ts_code", "date"], how="left")

        # 合并北向资金净流入（市场级）
        if moneyflow_hsgt is not None and not moneyflow_hsgt.empty:
            if "trade_date" in moneyflow_hsgt.columns:
                moneyflow_hsgt = moneyflow_hsgt.rename(columns={"trade_date": "date"})

            # 北向资金净流入
            flow_cols = ["date", "north_money", "north_net"]
            flow_cols = [c for c in flow_cols if c in moneyflow_hsgt.columns]
            if flow_cols:
                flow_subset = moneyflow_hsgt[flow_cols].copy()
                if "north_money" in flow_subset.columns:
                    flow_subset["northbound_net_flow"] = flow_subset["north_money"]
                elif "north_net" in flow_subset.columns:
                    flow_subset["northbound_net_flow"] = flow_subset["north_net"]
                df = df.merge(flow_subset[["date", "northbound_net_flow"]], on="date", how="left")

        # 计算北向资金滚动特征
        if "northbound_hold_ratio" in df.columns:
            df = df.sort_values(["ts_code", "date"])

            for window in self.northbound_windows:
                # 持股比例均值
                df[f"northbound_hold_ratio_{window}d_mean"] = df.groupby("ts_code")["northbound_hold_ratio"].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                # 持股比例变化
                df[f"northbound_hold_ratio_{window}d_change"] = df.groupby("ts_code")[
                    "northbound_hold_ratio"
                ].transform(lambda x: x.diff(window))

            # 北向资金趋势（持股比例是否在增加）
            df["northbound_trend"] = df.groupby("ts_code")["northbound_hold_ratio"].transform(
                lambda x: x.diff(5) / x.shift(5).replace(0, np.nan)
            )

        # 北向资金净流入滚动统计
        if "northbound_net_flow" in df.columns:
            for window in [5, 20]:
                df[f"northbound_net_flow_{window}d_sum"] = df.groupby("ts_code")["northbound_net_flow"].transform(
                    lambda x: x.rolling(window, min_periods=1).sum()
                )
                df[f"northbound_net_flow_{window}d_mean"] = df.groupby("ts_code")["northbound_net_flow"].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )

        # 北向资金综合评分
        df["northbound_score"] = self._calculate_northbound_score(df)

        logger.info("北向资金特征计算完成")
        return df

    def _calculate_northbound_score(self, df: pd.DataFrame) -> pd.Series:
        """计算北向资金综合评分"""
        scores = pd.Series(0.0, index=df.index)

        # 持股比例加分（持股越高越好）
        if "northbound_hold_ratio" in df.columns:
            hold_ratio_zscore = df.groupby("ts_code")["northbound_hold_ratio"].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
            scores += hold_ratio_zscore.fillna(0) * 0.4

        # 持股趋势加分（增持为正）
        if "northbound_hold_ratio_5d_change" in df.columns:
            change_zscore = df.groupby("ts_code")["northbound_hold_ratio_5d_change"].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
            scores += change_zscore.fillna(0) * 0.3

        # 净流入加分
        if "northbound_net_flow_5d_sum" in df.columns:
            flow_zscore = df.groupby("ts_code")["northbound_net_flow_5d_sum"].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
            scores += flow_zscore.fillna(0) * 0.3

        return scores

    def calculate_main_flow_features(
        self,
        df: pd.DataFrame,
        moneyflow: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        计算主力资金特征

        Args:
            df: 股票数据DataFrame
            moneyflow: 个股资金流向数据（buy_elg_vol, sell_elg_vol 等）

        Returns:
            包含主力资金特征的DataFrame
        """
        df = df.copy()

        # 初始化主力资金字段
        df["main_net_inflow"] = np.nan
        df["main_net_ratio"] = np.nan

        if moneyflow is not None and not moneyflow.empty:
            # 确保日期格式一致
            if "trade_date" in moneyflow.columns:
                moneyflow = moneyflow.rename(columns={"trade_date": "date"})

            # 计算主力净流入
            flow_cols = ["ts_code", "date"]
            calc_cols = []

            if "buy_elg_vol" in moneyflow.columns and "sell_elg_vol" in moneyflow.columns:
                moneyflow["main_buy"] = moneyflow["buy_elg_vol"].fillna(0) + moneyflow.get("buy_lg_vol", 0)
                moneyflow["main_sell"] = moneyflow["sell_elg_vol"].fillna(0) + moneyflow.get("sell_lg_vol", 0)
                moneyflow["main_net_inflow"] = moneyflow["main_buy"] - moneyflow["main_sell"]
                calc_cols.extend(["main_net_inflow"])

            if "buy_elg_amount" in moneyflow.columns and "sell_elg_amount" in moneyflow.columns:
                moneyflow["main_buy_amt"] = moneyflow["buy_elg_amount"].fillna(0) + moneyflow.get("buy_lg_amount", 0)
                moneyflow["main_sell_amt"] = moneyflow["sell_elg_amount"].fillna(0) + moneyflow.get("sell_lg_amount", 0)
                moneyflow["main_net_inflow_amt"] = moneyflow["main_buy_amt"] - moneyflow["main_sell_amt"]
                calc_cols.extend(["main_net_inflow_amt"])

            # 合并到主数据
            merge_cols = flow_cols + calc_cols
            merge_cols = [c for c in merge_cols if c in moneyflow.columns]
            if len(merge_cols) >= 3:
                df = df.merge(moneyflow[merge_cols], on=["ts_code", "date"], how="left")

        # 计算主力资金滚动特征
        if "main_net_inflow" in df.columns:
            df = df.sort_values(["ts_code", "date"])

            for window in self.flow_windows:
                df[f"main_net_inflow_{window}d_sum"] = df.groupby("ts_code")["main_net_inflow"].transform(
                    lambda x: x.rolling(window, min_periods=1).sum()
                )
                df[f"main_net_inflow_{window}d_mean"] = df.groupby("ts_code")["main_net_inflow"].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )

        # 主力资金评分
        if "main_net_inflow_5d_sum" in df.columns:
            df["main_flow_score"] = df.groupby("ts_code")["main_net_inflow_5d_sum"].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
        else:
            df["main_flow_score"] = np.nan

        logger.info("主力资金特征计算完成")
        return df

    def calculate_all_features(
        self,
        df: pd.DataFrame,
        hk_hold: Optional[pd.DataFrame] = None,
        moneyflow_hsgt: Optional[pd.DataFrame] = None,
        moneyflow: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        计算所有资金流向特征

        Args:
            df: 股票数据DataFrame
            hk_hold: 港资持股数据
            moneyflow_hsgt: 北向资金净流入数据
            moneyflow: 个股资金流向数据

        Returns:
            包含所有特征的DataFrame
        """
        logger.info("开始计算资金流向特征...")

        # 检查必要的列
        required_cols = ["ts_code", "date"]
        for col in required_cols:
            if col not in df.columns:
                if col == "date" and "trade_date" in df.columns:
                    df["date"] = df["trade_date"]
                else:
                    raise ValueError(f"缺少必要的列: {col}")

        # 计算各类特征
        df = self.calculate_northbound_features(df, hk_hold, moneyflow_hsgt)
        df = self.calculate_main_flow_features(df, moneyflow)

        logger.info("资金流向特征计算完成")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(df)
        return self.calculate_all_features(df)

    def get_feature_names(self) -> List[str]:
        """返回生成的特征名称列表"""
        return [
            # 北向资金特征
            "northbound_hold_ratio",
            "northbound_hold_amount",
            "northbound_net_flow",
            "northbound_hold_ratio_5d_mean",
            "northbound_hold_ratio_20d_mean",
            "northbound_hold_ratio_5d_change",
            "northbound_hold_ratio_20d_change",
            "northbound_trend",
            "northbound_net_flow_5d_sum",
            "northbound_net_flow_20d_sum",
            "northbound_score",
            # 主力资金特征
            "main_net_inflow",
            "main_net_inflow_5d_sum",
            "main_net_inflow_20d_sum",
            "main_flow_score",
        ]


if __name__ == "__main__":
    # 测试资金流向特征提取
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
                }
            )

    df = pd.DataFrame(data)

    # 模拟港资持股数据
    hk_hold = pd.DataFrame(
        {
            "ts_code": ["600519.SH"] * len(dates),
            "trade_date": [d.strftime("%Y%m%d") for d in dates],
            "hold_ratio": np.random.uniform(0.01, 0.1, len(dates)),
            "hold_amount": np.random.uniform(1e8, 1e9, len(dates)),
        }
    )

    feature_extractor = FlowFeatures()
    df_with_features = feature_extractor.calculate_all_features(df, hk_hold=hk_hold)

    print("\n生成的特征列:")
    print([c for c in df_with_features.columns if c in feature_extractor.get_feature_names()])

    print("\n数据预览:")
    print(df_with_features[["ts_code", "date", "northbound_hold_ratio", "northbound_score"]].tail(10))
