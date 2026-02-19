#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强风险控制模块

新增功能：
1. 基于confidence的动态仓位调整
2. 个股ATR动态止损
3. 行业层面止损
4. 组合层面分档降仓
5. 单日冲击止损
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskControlConfig:
    """风险控制配置"""

    # 1. 动态仓位配置
    base_position: float = 0.6  # 基础仓位
    max_position: float = 1.0  # 最大仓位
    min_position: float = 0.3  # 最小仓位
    confidence_multiplier: float = 1.0  # confidence乘数

    # 2. ATR止损配置
    atr_stop_loss_multiplier: float = 2.0  # ATR止损倍数
    atr_period: int = 14  # ATR计算周期
    enable_atr_stop: bool = True

    # 3. 行业止损配置
    industry_drawdown_threshold: float = -0.15  # 行业回撤阈值（-15%）
    enable_industry_stop: bool = True

    # 4. 组合分档降仓配置
    drawdown_tiers: List[Tuple[float, float]] = None  # [(回撤阈值, 仓位比例)]
    enable_tiered_drawdown: bool = True

    # 5. 单日冲击止损配置
    daily_shock_threshold: float = -0.03  # 单日跌幅阈值（-3%）
    daily_shock_position_cut: float = 0.5  # 触发后降仓比例（降至50%）
    enable_daily_shock_stop: bool = True

    # 6. 其他配置
    max_single_position: float = 0.10  # 单只股票最大仓位（10%）
    max_industry_exposure: float = 0.30  # 单行业最大暴露（30%）

    def __post_init__(self):
        if self.drawdown_tiers is None:
            self.drawdown_tiers = [
                (-0.10, 0.80),  # -10%回撤，降至80%仓位
                (-0.12, 0.60),  # -12%回撤，降至60%仓位
                (-0.15, 0.30),  # -15%回撤，降至30%仓位
            ]


class EnhancedRiskControl:
    """增强风险控制

    功能：
    1. confidence动态仓位：pos = base + (max-base) × confidence
    2. ATR动态止损：个股跌破ATR止损线自动卖出
    3. 行业止损：行业整体回撤超阈值清仓该行业
    4. 组合分档降仓：-10%/-12%/-15%分档降仓
    5. 单日冲击止损：单日跌幅超3%触发降仓
    """

    def __init__(self, config: Optional[RiskControlConfig] = None):
        self.config = config or RiskControlConfig()

        # 运行时状态
        self.portfolio_peak_value = 1.0
        self.portfolio_current_value = 1.0
        self.industry_peak_values: Dict[str, float] = {}
        self.stock_entry_prices: Dict[str, float] = {}
        self.stop_loss_events: List[Dict] = []

    def compute_dynamic_position(
        self,
        confidence: float,
        current_drawdown: Optional[float] = None,
        daily_return: Optional[float] = None,
        trade_date: Optional[str] = None,
    ) -> float:
        """计算动态仓位

        公式：pos = base + (max - base) × confidence

        Args:
            confidence: 趋势模型输出的置信度（0-1）
            current_drawdown: 当前回撤（可选）
            daily_return: 当日收益率（可选，用于单日冲击检测）
            trade_date: 交易日期（可选，用于事件记录）

        Returns:
            目标仓位（0-1）
        """
        # 1. 基于confidence计算基础仓位
        confidence = np.clip(confidence, 0.0, 1.0)
        base_position = (
            self.config.base_position
            + (self.config.max_position - self.config.base_position) * confidence * self.config.confidence_multiplier
        )

        # 2. 应用组合分档降仓
        if self.config.enable_tiered_drawdown and current_drawdown is not None:
            for threshold, position_ratio in sorted(self.config.drawdown_tiers):
                if current_drawdown <= threshold:
                    base_position *= position_ratio
                    self.stop_loss_events.append(
                        {
                            "date": trade_date or str(pd.Timestamp.now().date()),
                            "type": "TIERED_DRAWDOWN",
                            "drawdown": current_drawdown,
                            "threshold": threshold,
                            "position_ratio": position_ratio,
                        }
                    )
                    logger.info(f"组合回撤{current_drawdown:.2%}触发分档降仓，" f"仓位调整至{position_ratio:.0%}")
                    break

        # 3. 应用单日冲击止损
        if self.config.enable_daily_shock_stop and daily_return is not None:
            if daily_return <= self.config.daily_shock_threshold:
                base_position *= self.config.daily_shock_position_cut
                self.stop_loss_events.append(
                    {
                        "date": trade_date or str(pd.Timestamp.now().date()),
                        "type": "DAILY_SHOCK",
                        "daily_return": daily_return,
                        "threshold": self.config.daily_shock_threshold,
                        "position_cut": self.config.daily_shock_position_cut,
                    }
                )
                logger.warning(
                    f"单日跌幅{daily_return:.2%}触发冲击止损，" f"仓位降至{self.config.daily_shock_position_cut:.0%}"
                )

        # 4. 限制在最小/最大仓位之间
        final_position = np.clip(base_position, self.config.min_position, self.config.max_position)

        return final_position

    def compute_atr_stop_loss(
        self,
        prices: pd.Series,
        highs: pd.Series,
        lows: pd.Series,
        entry_price: float,
    ) -> float:
        """计算ATR动态止损价格

        Args:
            prices: 收盘价序列
            highs: 最高价序列
            lows: 最低价序列
            entry_price: 入场价格

        Returns:
            止损价格
        """
        if len(prices) < self.config.atr_period:
            # 数据不足，使用固定止损（-8%）
            return entry_price * 0.92

        # 计算ATR
        tr1 = highs - lows
        tr2 = abs(highs - prices.shift(1))
        tr3 = abs(lows - prices.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.config.atr_period).mean().iloc[-1]

        # 止损价 = 入场价 - ATR × 倍数
        stop_loss_price = entry_price - atr * self.config.atr_stop_loss_multiplier

        return stop_loss_price

    def check_stock_stop_loss(
        self,
        stock_data: pd.DataFrame,
        holdings: Dict[str, Dict],
    ) -> List[str]:
        """检查个股止损

        Args:
            stock_data: 股票数据（包含close/high/low）
            holdings: 当前持仓 {ts_code: {'entry_price': float, 'weight': float}}

        Returns:
            需要止损的股票代码列表
        """
        if not self.config.enable_atr_stop:
            return []

        stop_loss_stocks = []

        for ts_code, holding in holdings.items():
            if ts_code not in stock_data.index:
                continue

            stock = stock_data.loc[ts_code]
            entry_price = holding.get("entry_price", stock["close"])

            # 计算止损价
            if "high" in stock_data.columns and "low" in stock_data.columns:
                # 获取历史数据计算ATR
                hist_data = stock_data[stock_data.index == ts_code].tail(self.config.atr_period + 1)
                if len(hist_data) >= self.config.atr_period:
                    stop_price = self.compute_atr_stop_loss(
                        hist_data["close"],
                        hist_data["high"],
                        hist_data["low"],
                        entry_price,
                    )
                else:
                    # 数据不足，使用固定止损
                    stop_price = entry_price * 0.92
            else:
                # 无高低价数据，使用固定止损
                stop_price = entry_price * 0.92

            # 检查是否触发止损
            current_price = stock["close"]
            if current_price <= stop_price:
                stop_loss_stocks.append(ts_code)
                loss_pct = (current_price - entry_price) / entry_price

                self.stop_loss_events.append(
                    {
                        "date": stock.get("trade_date", pd.Timestamp.now()),
                        "ts_code": ts_code,
                        "type": "ATR_STOP",
                        "entry_price": entry_price,
                        "stop_price": stop_price,
                        "current_price": current_price,
                        "loss_pct": loss_pct,
                    }
                )

                logger.warning(
                    f"个股{ts_code}触发ATR止损：入场价{entry_price:.2f}，"
                    f"止损价{stop_price:.2f}，当前价{current_price:.2f}，"
                    f"亏损{loss_pct:.2%}"
                )

        return stop_loss_stocks

    def check_industry_stop_loss(
        self,
        industry_returns: Dict[str, float],
        trade_date: Optional[str] = None,
    ) -> List[str]:
        """检查行业止损

        Args:
            industry_returns: 行业累计收益率 {industry: return}
            trade_date: 交易日期（可选，用于事件记录）

        Returns:
            需要清仓的行业列表
        """
        if not self.config.enable_industry_stop:
            return []

        stop_loss_industries = []

        for industry, cum_return in industry_returns.items():
            # 更新行业峰值
            if industry not in self.industry_peak_values:
                self.industry_peak_values[industry] = 1.0

            current_value = 1.0 + cum_return
            self.industry_peak_values[industry] = max(self.industry_peak_values[industry], current_value)

            # 计算行业回撤
            drawdown = current_value / self.industry_peak_values[industry] - 1.0

            # 检查是否触发止损
            if drawdown <= self.config.industry_drawdown_threshold:
                stop_loss_industries.append(industry)

                self.stop_loss_events.append(
                    {
                        "date": trade_date or str(pd.Timestamp.now().date()),
                        "industry": industry,
                        "type": "INDUSTRY_STOP",
                        "peak_value": self.industry_peak_values[industry],
                        "current_value": current_value,
                        "drawdown": drawdown,
                    }
                )

                logger.warning(
                    f"行业{industry}触发止损：峰值{self.industry_peak_values[industry]:.2f}，"
                    f"当前{current_value:.2f}，回撤{drawdown:.2%}"
                )

        return stop_loss_industries

    def update_portfolio_value(self, current_value: float):
        """更新组合价值（用于计算回撤）"""
        self.portfolio_current_value = current_value
        self.portfolio_peak_value = max(self.portfolio_peak_value, current_value)

    def get_current_drawdown(self) -> float:
        """获取当前回撤"""
        if self.portfolio_peak_value <= 0:
            return 0.0
        return self.portfolio_current_value / self.portfolio_peak_value - 1.0

    def apply_position_limits(
        self,
        weights: pd.Series,
        industries: Optional[pd.Series] = None,
    ) -> pd.Series:
        """应用仓位限制

        Args:
            weights: 原始权重
            industries: 行业分类（可选）

        Returns:
            调整后的权重
        """
        adjusted_weights = weights.copy()

        # 1. 单只股票仓位限制
        adjusted_weights = adjusted_weights.clip(upper=self.config.max_single_position)

        # 2. 行业仓位限制
        if industries is not None and self.config.max_industry_exposure < 1.0:
            # 确保索引对齐
            if not adjusted_weights.index.equals(industries.index):
                industries = industries.reindex(adjusted_weights.index)

            for industry in industries.unique():
                if pd.isna(industry):
                    continue
                industry_mask = industries == industry
                industry_weight = adjusted_weights[industry_mask].sum()

                if industry_weight > self.config.max_industry_exposure:
                    # 按比例缩减该行业的权重
                    scale_factor = self.config.max_industry_exposure / industry_weight
                    adjusted_weights[industry_mask] *= scale_factor

        # 3. 重新归一化
        total_weight = adjusted_weights.sum()
        if total_weight > 0:
            adjusted_weights = adjusted_weights / total_weight

        return adjusted_weights

    def get_stop_loss_report(self) -> pd.DataFrame:
        """获取止损事件报告"""
        if not self.stop_loss_events:
            return pd.DataFrame()

        return pd.DataFrame(self.stop_loss_events)

    def get_stop_loss_summary(self) -> Dict:
        """获取止损事件统计摘要"""
        if not self.stop_loss_events:
            return {"total_events": 0}

        df = pd.DataFrame(self.stop_loss_events)
        summary: Dict = {
            "total_events": len(df),
            "by_type": df["type"].value_counts().to_dict(),
        }

        # ATR止损统计
        atr_events = df[df["type"] == "ATR_STOP"]
        if not atr_events.empty:
            summary["atr_stop"] = {
                "count": len(atr_events),
                "avg_loss": float(atr_events["loss_pct"].mean()),
                "max_loss": float(atr_events["loss_pct"].min()),
                "stocks_affected": atr_events["ts_code"].nunique(),
            }

        # 行业止损统计
        ind_events = df[df["type"] == "INDUSTRY_STOP"]
        if not ind_events.empty:
            summary["industry_stop"] = {
                "count": len(ind_events),
                "avg_drawdown": float(ind_events["drawdown"].mean()),
                "industries_affected": ind_events["industry"].nunique(),
            }

        # 分档降仓统计
        tier_events = df[df["type"] == "TIERED_DRAWDOWN"]
        if not tier_events.empty:
            summary["tiered_drawdown"] = {
                "count": len(tier_events),
                "avg_drawdown": float(tier_events["drawdown"].mean()),
            }

        # 单日冲击止损统计
        shock_events = df[df["type"] == "DAILY_SHOCK"]
        if not shock_events.empty:
            summary["daily_shock"] = {
                "count": len(shock_events),
                "avg_daily_return": float(shock_events["daily_return"].mean()),
            }

        return summary
