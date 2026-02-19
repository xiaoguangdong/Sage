#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
交易成本模型

包含三种成本：
1. 固定成本：佣金、印花税等（单边）
2. 滑点成本：固定滑点 + 波动率滑点
3. 市场冲击成本：基于Almgren-Chriss简化模型

参考文献：
- Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions.
- Kissell, R., & Glantz, M. (2013). Optimal Trading Strategies.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml


@dataclass
class CostModelConfig:
    """交易成本模型配置"""

    # 1. 固定成本（单边）
    commission_rate: float = 0.0003  # 佣金费率（万三）
    stamp_tax_rate: float = 0.001  # 印花税（卖出时收取）
    transfer_fee_rate: float = 0.00002  # 过户费

    # 2. 滑点成本
    fixed_slippage_bps: float = 1.0  # 固定滑点（基点，1bp=0.01%）
    volatility_slippage_factor: float = 0.1  # 波动率滑点系数（降低）

    # 3. 市场冲击成本（Almgren-Chriss简化）
    market_impact_factor: float = 0.05  # 市场冲击系数（降低）
    temporary_impact_decay: float = 0.3  # 临时冲击衰减系数（降低）

    # 4. 流动性约束
    max_participation_rate: float = 0.05  # 最大成交量占比（5%）
    min_daily_volume: float = 1_000_000  # 最小日成交额（元）

    @classmethod
    def from_yaml(cls, path: str = "config/app/backtest_costs.yaml", profile: str = "default") -> "CostModelConfig":
        """从 YAML 配置文件加载

        Args:
            path: 配置文件路径（相对于项目根目录或绝对路径）
            profile: 配置档位（default / conservative / aggressive）
        """
        config_path = Path(path)
        if not config_path.is_absolute():
            config_path = Path(__file__).resolve().parents[2] / path

        with open(config_path, "r", encoding="utf-8") as f:
            all_cfg = yaml.safe_load(f)

        cfg = all_cfg.get(profile, {})
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})

    @property
    def total_fixed_cost_buy(self) -> float:
        """买入固定成本"""
        return self.commission_rate + self.transfer_fee_rate

    @property
    def total_fixed_cost_sell(self) -> float:
        """卖出固定成本"""
        return self.commission_rate + self.stamp_tax_rate + self.transfer_fee_rate


class TradingCostModel:
    """交易成本模型

    计算完整的交易成本，包括：
    - 固定成本（佣金、印花税、过户费）
    - 滑点成本（固定 + 波动率相关）
    - 市场冲击成本（永久冲击 + 临时冲击）
    """

    def __init__(self, config: Optional[CostModelConfig] = None):
        self.config = config or CostModelConfig()

    def calculate_total_cost(
        self,
        trade_value: float,
        is_buy: bool,
        volatility: float = 0.02,
        daily_volume: float = 10_000_000,
        avg_price: float = 10.0,
    ) -> Dict[str, float]:
        """计算总交易成本

        Args:
            trade_value: 交易金额（元）
            is_buy: 是否买入
            volatility: 日波动率（默认2%）
            daily_volume: 日成交额（元，默认1000万）
            avg_price: 平均价格（元，默认10元）

        Returns:
            成本明细字典：
            {
                'fixed_cost': 固定成本,
                'slippage_cost': 滑点成本,
                'market_impact_cost': 市场冲击成本,
                'total_cost': 总成本,
                'total_cost_rate': 总成本率
            }
        """
        # 1. 固定成本
        if is_buy:
            fixed_cost_rate = self.config.total_fixed_cost_buy
        else:
            fixed_cost_rate = self.config.total_fixed_cost_sell

        fixed_cost = trade_value * fixed_cost_rate

        # 2. 滑点成本
        slippage_cost = self._calculate_slippage(trade_value, volatility)

        # 3. 市场冲击成本
        market_impact_cost = self._calculate_market_impact(trade_value, daily_volume, volatility, avg_price)

        # 4. 总成本
        total_cost = fixed_cost + slippage_cost + market_impact_cost
        total_cost_rate = total_cost / trade_value if trade_value > 0 else 0.0

        return {
            "fixed_cost": fixed_cost,
            "slippage_cost": slippage_cost,
            "market_impact_cost": market_impact_cost,
            "total_cost": total_cost,
            "total_cost_rate": total_cost_rate,
        }

    def _calculate_slippage(self, trade_value: float, volatility: float) -> float:
        """计算滑点成本

        滑点 = 固定滑点 + 波动率滑点

        Args:
            trade_value: 交易金额
            volatility: 日波动率

        Returns:
            滑点成本（元）
        """
        # 固定滑点（基点转换为比例）
        fixed_slippage = trade_value * (self.config.fixed_slippage_bps / 10000)

        # 波动率滑点（波动率越高，滑点越大）
        volatility_slippage = trade_value * volatility * self.config.volatility_slippage_factor

        return fixed_slippage + volatility_slippage

    def _calculate_market_impact(
        self,
        trade_value: float,
        daily_volume: float,
        volatility: float,
        avg_price: float,
    ) -> float:
        """计算市场冲击成本（Almgren-Chriss简化模型）

        市场冲击 = 永久冲击 + 临时冲击

        永久冲击：交易后价格永久性变化
        临时冲击：交易过程中的临时价格压力

        Args:
            trade_value: 交易金额
            daily_volume: 日成交额
            volatility: 日波动率
            avg_price: 平均价格

        Returns:
            市场冲击成本（元）
        """
        if daily_volume <= 0:
            # 无成交量数据，使用默认冲击
            return trade_value * 0.001

        # 参与率（交易量占日成交量的比例）
        participation_rate = trade_value / daily_volume

        # 限制最大参与率
        participation_rate = min(participation_rate, self.config.max_participation_rate)

        # 永久冲击（与参与率和波动率成正比）
        # 公式：permanent_impact = α × σ × (Q/V)^0.5
        permanent_impact = self.config.market_impact_factor * volatility * np.sqrt(participation_rate) * trade_value

        # 临时冲击（交易过程中的价格压力，会部分恢复）
        # 公式：temporary_impact = β × σ × (Q/V)
        temporary_impact = (
            self.config.market_impact_factor
            * self.config.temporary_impact_decay
            * volatility
            * participation_rate
            * trade_value
        )

        return permanent_impact + temporary_impact

    def calculate_turnover_cost(
        self,
        turnover: float,
        portfolio_value: float,
        market_data: Optional[pd.DataFrame] = None,
    ) -> float:
        """计算换手成本（简化版，用于回测）

        Args:
            turnover: 换手率（0-2之间，表示权重变化的绝对值之和）
                     例如：0.5表示50%的仓位发生变化
            portfolio_value: 组合价值
            market_data: 市场数据（可选，包含波动率和成交量）

        Returns:
            换手成本率（相对于组合价值）
        """
        if turnover <= 0:
            return 0.0

        # 交易金额（换手率已经是单边，直接乘以组合价值）
        trade_value = turnover * portfolio_value

        # 默认市场参数
        avg_volatility = 0.02  # 2%日波动率
        avg_daily_volume = 10_000_000  # 1000万日成交额
        avg_price = 10.0

        # 如果有市场数据，使用实际参数
        if market_data is not None and not market_data.empty:
            if "volatility" in market_data.columns:
                avg_volatility = market_data["volatility"].mean()
            if "amount" in market_data.columns:
                avg_daily_volume = market_data["amount"].mean()
            if "close" in market_data.columns:
                avg_price = market_data["close"].mean()

        # 假设买卖各占一半
        buy_cost = self.calculate_total_cost(
            trade_value / 2,
            is_buy=True,
            volatility=avg_volatility,
            daily_volume=avg_daily_volume,
            avg_price=avg_price,
        )

        sell_cost = self.calculate_total_cost(
            trade_value / 2,
            is_buy=False,
            volatility=avg_volatility,
            daily_volume=avg_daily_volume,
            avg_price=avg_price,
        )

        total_cost = buy_cost["total_cost"] + sell_cost["total_cost"]
        cost_rate = total_cost / portfolio_value

        return cost_rate

    def check_liquidity_constraint(
        self,
        trade_value: float,
        daily_volume: float,
    ) -> bool:
        """检查流动性约束

        Args:
            trade_value: 交易金额
            daily_volume: 日成交额

        Returns:
            是否满足流动性约束
        """
        # 检查最小成交额
        if daily_volume < self.config.min_daily_volume:
            return False

        # 检查参与率
        participation_rate = trade_value / daily_volume if daily_volume > 0 else 1.0
        if participation_rate > self.config.max_participation_rate:
            return False

        return True


def create_default_cost_model() -> TradingCostModel:
    """创建默认成本模型（优先从配置文件加载）"""
    try:
        config = CostModelConfig.from_yaml(profile="default")
    except FileNotFoundError:
        config = CostModelConfig()
    return TradingCostModel(config)


def create_conservative_cost_model() -> TradingCostModel:
    """创建保守成本模型（较高成本估计）"""
    try:
        config = CostModelConfig.from_yaml(profile="conservative")
    except FileNotFoundError:
        config = CostModelConfig(
            commission_rate=0.0003,
            stamp_tax_rate=0.001,
            transfer_fee_rate=0.00002,
            fixed_slippage_bps=2.0,
            volatility_slippage_factor=0.5,
            market_impact_factor=0.1,
            temporary_impact_decay=0.5,
            max_participation_rate=0.05,
            min_daily_volume=1_000_000,
        )
    return TradingCostModel(config)


def create_aggressive_cost_model() -> TradingCostModel:
    """创建激进成本模型（乐观估计，用于对比）"""
    try:
        config = CostModelConfig.from_yaml(profile="aggressive")
    except FileNotFoundError:
        config = CostModelConfig(
            commission_rate=0.0001,
            stamp_tax_rate=0.001,
            transfer_fee_rate=0.00002,
            fixed_slippage_bps=1.0,
            volatility_slippage_factor=0.3,
            market_impact_factor=0.05,
            temporary_impact_decay=0.3,
            max_participation_rate=0.10,
            min_daily_volume=500_000,
        )
    return TradingCostModel(config)
