"""
组合构建模块
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PortfolioConstruction:
    """组合构建类"""

    def __init__(self, config: dict = None):
        """
        初始化组合构建器

        Args:
            config: 配置字典
        """
        self.config = config or {}

        # 默认配置
        self.default_config = {
            "method": "equal_weight",  # 等权重配置
            "max_positions": 50,  # 最大持仓数
            "min_weight": 0.01,  # 最小权重
            "max_weight": 0.05,  # 最大权重
            "sector_limits": None,  # 行业限制
        }

        # 合并配置
        self.config = {**self.default_config, **self.config}

    def construct_portfolio(self, df_ranked: pd.DataFrame, trend_state: int = 1) -> pd.DataFrame:
        """
        构建组合

        Args:
            df_ranked: 排序后的股票数据，包含'rank_score'和'rank'列
            trend_state: 趋势状态（0=RISK_OFF, 1=震荡, 2=RISK_ON）

        Returns:
            包含组合权重的DataFrame
        """
        logger.info(f"构建组合，趋势状态: {trend_state}")

        if trend_state == 0:
            # RISK_OFF状态，空仓或低仓位
            logger.info("RISK_OFF状态，构建低风险组合")
            return self._construct_low_risk_portfolio(df_ranked)
        elif trend_state == 2:
            # RISK_ON状态，高仓位
            logger.info("RISK_ON状态，构建高收益组合")
            return self._construct_high_risk_portfolio(df_ranked)
        else:
            # 震荡状态，中等仓位
            logger.info("震荡状态，构建中等风险组合")
            return self._construct_balanced_portfolio(df_ranked)

    def _construct_equal_weight_portfolio(self, df_ranked: pd.DataFrame, n_stocks: int = None) -> pd.DataFrame:
        """
        构建等权重组合

        Args:
            df_ranked: 排序后的股票数据
            n_stocks: 选股数量

        Returns:
            包含权重的DataFrame
        """
        if n_stocks is None:
            n_stocks = self.config["max_positions"]

        # 选择前n_stocks只股票
        df_selected = df_ranked.head(n_stocks).copy()

        # 等权重分配
        weight = 1.0 / len(df_selected)
        df_selected["weight"] = weight

        logger.info(f"等权重组合，选择{len(df_selected)}只股票，每只权重: {weight:.2%}")

        return df_selected

    def _construct_low_risk_portfolio(self, df_ranked: pd.DataFrame) -> pd.DataFrame:
        """
        构建低风险组合（RISK_OFF状态）

        Args:
            df_ranked: 排序后的股票数据

        Returns:
            包含权重的DataFrame
        """
        # RISK_OFF状态，选择少量高排名股票，低仓位
        n_stocks = min(10, self.config["max_positions"])
        df_selected = self._construct_equal_weight_portfolio(df_ranked, n_stocks)

        # 降低仓位至30%
        df_selected["weight"] = df_selected["weight"] * 0.3

        logger.info(f"低风险组合，选择{len(df_selected)}只股票，总仓位: {df_selected['weight'].sum():.2%}")

        return df_selected

    def _construct_balanced_portfolio(self, df_ranked: pd.DataFrame) -> pd.DataFrame:
        """
        构建中等风险组合（震荡状态）

        Args:
            df_ranked: 排序后的股票数据

        Returns:
            包含权重的DataFrame
        """
        # 震荡状态，选择中等数量股票，中等仓位
        n_stocks = min(30, self.config["max_positions"])
        df_selected = self._construct_equal_weight_portfolio(df_ranked, n_stocks)

        # 中等仓位60%
        df_selected["weight"] = df_selected["weight"] * 0.6

        logger.info(f"中等风险组合，选择{len(df_selected)}只股票，总仓位: {df_selected['weight'].sum():.2%}")

        return df_selected

    def _construct_high_risk_portfolio(self, df_ranked: pd.DataFrame) -> pd.DataFrame:
        """
        构建高风险组合（RISK_ON状态）

        Args:
            df_ranked: 排序后的股票数据

        Returns:
            包含权重的DataFrame
        """
        # RISK_ON状态，选择最多股票，高仓位
        n_stocks = self.config["max_positions"]
        df_selected = self._construct_equal_weight_portfolio(df_ranked, n_stocks)

        # 高仓位90%
        df_selected["weight"] = df_selected["weight"] * 0.9

        logger.info(f"高风险组合，选择{len(df_selected)}只股票，总仓位: {df_selected['weight'].sum():.2%}")

        return df_selected

    def apply_sector_limits(self, df_portfolio: pd.DataFrame, sector_limits: Dict[str, float]) -> pd.DataFrame:
        """
        应用行业限制

        Args:
            df_portfolio: 组合数据，需要包含'sector'列
            sector_limits: 行业限制字典，如{'金融': 0.3, '科技': 0.4}

        Returns:
            调整后的组合
        """
        if "sector" not in df_portfolio.columns:
            logger.warning("组合数据不包含'sector'列，无法应用行业限制")
            return df_portfolio

        # 计算每个行业的当前权重
        sector_weights = df_portfolio.groupby("sector")["weight"].sum()

        # 检查是否超过限制
        for sector, limit in sector_limits.items():
            if sector in sector_weights and sector_weights[sector] > limit:
                logger.warning(f"行业'{sector}'权重{sector_weights[sector]:.2%}超过限制{limit:.2%}")
                # TODO: 实现权重调整逻辑

        return df_portfolio

    def calculate_portfolio_returns(self, df_portfolio: pd.DataFrame, df_prices: pd.DataFrame) -> pd.Series:
        """
        计算组合收益

        Args:
            df_portfolio: 组合数据，包含'code'和'weight'列
            df_prices: 价格数据，包含'code'和'return'列

        Returns:
            组合收益序列
        """
        # 合并权重和收益
        df_merged = df_portfolio[["code", "weight"]].merge(df_prices[["code", "return"]], on="code", how="inner")

        # 计算加权收益
        portfolio_return = (df_merged["weight"] * df_merged["return"]).sum()

        return portfolio_return


if __name__ == "__main__":
    # 测试组合构建
    logging.basicConfig(level=logging.INFO)

    # 创建测试数据
    np.random.seed(42)
    stock_codes = ["sh.600000", "sh.600004", "sh.600006", "sh.600007", "sh.600008"]

    data = []
    for i, code in enumerate(stock_codes):
        data.append(
            {
                "code": code,
                "rank_score": np.random.rand(),
                "rank": i + 1,
                "sector": ["金融", "金融", "科技", "科技", "消费"][i],
            }
        )

    df_ranked = pd.DataFrame(data)

    # 测试组合构建
    print("测试组合构建...")
    constructor = PortfolioConstruction()

    # 测试不同趋势状态
    for state, state_name in [(0, "RISK_OFF"), (1, "震荡"), (2, "RISK_ON")]:
        print(f"\n测试{state_name}状态:")
        portfolio = constructor.construct_portfolio(df_ranked, state)
        print(portfolio[["code", "rank", "weight"]])
