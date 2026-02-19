"""
股票池筛选
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Universe:
    """股票池筛选器"""

    def __init__(self):
        """初始化股票池筛选器"""
        self.st_list = []  # ST股票列表
        self.suspended_list = []  # 停牌股票列表

    def filter_stocks(
        self,
        df: pd.DataFrame,
        exclude_st: bool = True,
        exclude_suspended: bool = True,
        min_turnover: Optional[float] = None,
        min_market_cap: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        筛选股票池

        Args:
            df: 包含股票数据的DataFrame
            exclude_st: 是否剔除ST股票
            exclude_suspended: 是否剔除停牌股票
            min_turnover: 最小换手率（可选）
            min_market_cap: 最小市值（可选）

        Returns:
            筛选后的DataFrame
        """
        logger.info(f"原始股票数: {df['stock'].nunique()}")

        # 剔除ST股票
        if exclude_st and "is_st" in df.columns:
            before_count = len(df)
            df = df[not df["is_st"]]
            logger.info(f"剔除ST股票: {before_count - len(df)} 只")

        # 剔除停牌股票
        if exclude_suspended and "is_suspended" in df.columns:
            before_count = len(df)
            df = df[not df["is_suspended"]]
            logger.info(f"剔除停牌股票: {before_count - len(df)} 只")

        # 换手率过滤
        if min_turnover is not None and "turnover" in df.columns:
            before_count = len(df)
            df = df[df["turnover"] >= min_turnover]
            logger.info(f"换手率过滤: {before_count - len(df)} 只")

        # 市值过滤
        if min_market_cap is not None and "market_cap" in df.columns:
            before_count = len(df)
            df = df[df["market_cap"] >= min_market_cap]
            logger.info(f"市值过滤: {before_count - len(df)} 只")

        logger.info(f"筛选后股票数: {df['stock'].nunique()}")

        return df

    def get_available_stocks(self, df: pd.DataFrame) -> List[str]:
        """
        获取可用股票列表

        Args:
            df: 包含股票数据的DataFrame

        Returns:
            股票代码列表
        """
        return df["stock"].unique().tolist()


if __name__ == "__main__":
    # 测试股票池筛选
    logging.basicConfig(level=logging.INFO)

    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    stocks = ["sh.600000", "sh.600001", "sh.600002", "sh.600003", "sh.600004"]

    data = []
    for stock in stocks:
        for date in dates:
            data.append(
                {
                    "date": date,
                    "stock": stock,
                    "close": np.random.uniform(10, 100),
                    "is_st": np.random.choice([True, False], p=[0.1, 0.9]),
                    "is_suspended": np.random.choice([True, False], p=[0.05, 0.95]),
                    "turnover": np.random.uniform(0.5, 5),
                    "market_cap": np.random.uniform(50, 500) * 1e8,
                }
            )

    df = pd.DataFrame(data)

    universe = Universe()
    filtered_df = universe.filter_stocks(
        df, exclude_st=True, exclude_suspended=True, min_turnover=1.0, min_market_cap=100e8
    )

    print("\n筛选结果:")
    print(filtered_df.head())
