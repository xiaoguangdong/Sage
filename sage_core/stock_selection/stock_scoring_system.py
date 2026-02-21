"""
股票评分和买卖点判断系统
基于docs/market_tendcy_qa.md的量化规则

功能：
1. 股票综合评分（100分制）
2. 买入信号判断
3. 持有信号判断
4. 卖出信号判断
5. 风险管理和仓位控制
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sage_core.utils.logging_utils import format_task_summary, setup_logging

logger = logging.getLogger(__name__)


class StockScoringSystem:
    """股票评分系统（100分制）"""

    def __init__(self, industry_data: pd.DataFrame = None, fundamental_data: pd.DataFrame = None):
        """
        初始化评分系统

        Args:
            industry_data: 行业数据用于分位数计算
            fundamental_data: 财务基本面数据
        """
        self.industry_data = industry_data
        self.fundamental_data = fundamental_data

    def calculate_score(self, df: pd.DataFrame, stock_code: str = None) -> Dict:
        """
        计算股票综合评分

        Args:
            df: 股票数据DataFrame，必须包含OHLCV数据
            stock_code: 股票代码（用于获取基本面数据）

        Returns:
            包含总分和分项得分的字典
        """
        df = df.copy()
        df = self._calculate_indicators(df)

        # 获取基本面数据
        fundamentals = self._get_stock_fundamentals(stock_code) if stock_code else None

        # 计算各维度得分
        fundamental_score = self._calculate_fundamental_score(df, fundamentals)
        technical_score = self._calculate_technical_score(df)
        risk_score = self._calculate_risk_score(df)

        # 计算总分（0-100）
        total_score = fundamental_score + technical_score + risk_score

        # 评级
        rating = self._get_rating(total_score)

        result = {
            "total_score": total_score,
            "rating": rating["grade"],
            "recommendation": rating["recommendation"],
            "expected_return": rating["expected_return"],
            "fundamental_score": fundamental_score,
            "technical_score": technical_score,
            "risk_score": risk_score,
            "fundamental_details": self._get_fundamental_details(fundamentals),
            "technical_details": self._get_technical_details(df),
            "risk_details": self._get_risk_details(df),
            "key_metrics": self._get_key_metrics(df),
        }

        logger.info(f"股票{stock_code}评分完成: 总分={total_score:.2f}, 评级={rating['grade']}")

        return result

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = df.copy()

        # 移动平均线
        df["ma5"] = df["close"].rolling(5).mean()
        df["ma10"] = df["close"].rolling(10).mean()
        df["ma20"] = df["close"].rolling(20).mean()
        df["ma60"] = df["close"].rolling(60).mean()
        df["ma120"] = df["close"].rolling(120).mean()
        df["ma250"] = df["close"].rolling(250).mean()

        # 均线斜率
        df["ma5_slope"] = df["ma5"].diff(5)
        df["ma20_slope"] = df["ma20"].diff(5)
        df["ma60_slope"] = df["ma60"].diff(5)

        # 价格位置
        df["price_vs_ma20"] = df["close"] / df["ma20"] - 1
        df["price_vs_ma60"] = df["close"] / df["ma60"] - 1
        df["price_vs_ma120"] = df["close"] / df["ma120"] - 1

        # RSI
        df["rsi14"] = self._calculate_rsi(df["close"], 14)
        df["rsi30"] = self._calculate_rsi(df["close"], 30)

        # MACD
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # 布林带
        bb_middle = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        df["bb_upper"] = bb_middle + 2 * bb_std
        df["bb_lower"] = bb_middle - 2 * bb_std
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_middle

        # ATR（平均真实波幅）
        df["tr"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(abs(df["high"] - df["close"].shift(1)), abs(df["low"] - df["close"].shift(1))),
        )
        df["atr_14"] = df["tr"].rolling(14).mean()
        df["atr_20"] = df["tr"].rolling(20).mean()

        # 波动率
        df["volatility"] = df["close"].pct_change().rolling(20).std()
        df["volatility_ma"] = df["volatility"].rolling(60).median()

        # 动量
        df["momentum_5d"] = df["close"].pct_change(5)
        df["momentum_20d"] = df["close"].pct_change(20)
        df["momentum_60d"] = df["close"].pct_change(60)

        # 成交量指标
        df["volume_ma20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma20"]

        # 换手率
        df["turnover_rate"] = df["turn"] if "turn" in df.columns else 0

        # 最高最低价（120日）
        df["high_120"] = df["high"].rolling(120).max()
        df["low_120"] = df["low"].rolling(120).min()

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _get_stock_fundamentals(self, stock_code: str) -> Optional[Dict]:
        """
        获取股票基本面数据
        TODO: 集成Tushare API获取真实财务数据
        """
        if self.fundamental_data is not None:
            stock_fundamentals = self.fundamental_data[self.fundamental_data["code"] == stock_code]
            if not stock_fundamentals.empty:
                return stock_fundamentals.iloc[-1].to_dict()

        # 返回默认值（待实现真实数据获取）
        return {
            "roe": 0.10,  # 净资产收益率
            "roic": 0.08,  # 投入资本回报率
            "revenue_growth": 0.10,  # 收入增长率
            "eps_growth": 0.12,  # EPS增长率
            "pe": 20.0,  # 市盈率
            "pb": 2.5,  # 市净率
            "debt_ratio": 0.45,  # 资产负债率
            "cash_flow_ratio": 1.1,  # 经营现金流/净利润
            "dividend_ratio": 0.30,  # 分红率
        }

    def _calculate_fundamental_score(self, df: pd.DataFrame, fundamentals: Optional[Dict]) -> float:
        """
        计算基本面得分（50分）

        指标说明：
        1. 盈利能力（15分）: ROE, ROIC, 盈利稳定性
        2. 成长性（15分）: 收入增长、利润增长、增长质量
        3. 估值合理性（10分）: PE, PB, PEG
        4. 财务质量（10分）: 资产负债率、现金流质量、营运效率、分红能力

        数据来源：
        - Baostock: peTTM, pbMRQ (估值)
        - Tushare: profit/fin_indicator (盈利能力), growth (成长性), balancesheet (财务质量)
        """
        score = 0.0

        # 使用技术指标作为基本面代理（实际应该用真实财务数据）
        if fundamentals:
            # 1. 盈利能力（15分）
            # ROE评分
            roe = fundamentals.get("roe", 0.10)
            if roe > 0.20:
                score += 10
            elif roe > 0.15:
                score += 8
            elif roe > 0.10:
                score += 6
            elif roe > 0.05:
                score += 3

            # ROIC评分
            roic = fundamentals.get("roic", 0.08)
            if roic > 0.15:
                score += 5
            elif roic > 0.10:
                score += 3
            elif roic > 0.05:
                score += 1

            # 2. 成长性（15分）
            # 收入增长
            revenue_growth = fundamentals.get("revenue_growth", 0.10)
            if revenue_growth > 0.20:
                score += 6
            elif revenue_growth > 0.10:
                score += 4
            elif revenue_growth > 0.05:
                score += 2

            # 利润增长
            eps_growth = fundamentals.get("eps_growth", 0.12)
            if eps_growth > 0.15:
                score += 4
            elif eps_growth > 0.08:
                score += 3
            elif eps_growth > 0:
                score += 1

            # 增长质量（现金流增长 > 利润增长）
            # 需要Tushare的cashflow数据
            score += 5  # 默认分

            # 3. 估值合理性（10分）
            # PE估值（与行业比）
            pe = fundamentals.get("pe", df.get("peTTM", [20.0]).iloc[-1])
            if pe < 10:
                score += 3
            elif pe < 20:
                score += 2
            elif pe < 30:
                score += 1

            # PB估值
            pb = fundamentals.get("pb", df.get("pbMRQ", [2.5]).iloc[-1])
            if pb < 1.5:
                score += 3
            elif pb < 2.5:
                score += 2
            elif pb < 4.0:
                score += 1

            # PEG估值
            if eps_growth > 0:
                peg = pe / eps_growth / 100
                if peg < 0.8:
                    score += 4
                elif peg < 1.0:
                    score += 3
                elif peg < 1.2:
                    score += 2

            # 4. 财务质量（10分）
            # 资产负债率
            debt_ratio = fundamentals.get("debt_ratio", 0.45)
            if debt_ratio < 0.40:
                score += 3
            elif debt_ratio < 0.60:
                score += 2
            elif debt_ratio < 0.80:
                score += 1

            # 现金流质量
            cash_flow_ratio = fundamentals.get("cash_flow_ratio", 1.1)
            if cash_flow_ratio > 1.2:
                score += 3
            elif cash_flow_ratio > 0.8:
                score += 2
            elif cash_flow_ratio > 0:
                score += 1

            # 分红能力
            dividend_ratio = fundamentals.get("dividend_ratio", 0.30)
            if dividend_ratio > 0.30:
                score += 2
            elif dividend_ratio > 0:
                score += 1
        else:
            # 没有基本面数据时，用技术指标代理（降权）
            logger.warning("无基本面数据，使用技术指标代理（得分降低）")

            # 1. 盈利能力代理（基于价格趋势）
            price_strength = df["price_vs_ma20"].iloc[-1]
            if price_strength > 0.15:
                score += 7.5  # 原来是15分，减半
            elif price_strength > 0.05:
                score += 4.5
            elif price_strength > 0:
                score += 3

            # 2. 成长性代理（基于动量）
            momentum_20d = df["momentum_20d"].iloc[-1]
            if momentum_20d > 0.15:
                score += 7.5
            elif momentum_20d > 0.05:
                score += 4.5
            elif momentum_20d > 0:
                score += 2.5

            # 3. 估值代理（基于布林带）
            bb_position = df["bb_position"].iloc[-1]
            if 0.3 < bb_position < 0.7:
                score += 5
            elif 0.2 < bb_position < 0.8:
                score += 3.5
            else:
                score += 1.5

            # 4. 财务质量代理（基于波动率）
            volatility = df["volatility"].iloc[-1]
            if volatility < 0.015:
                score += 5
            elif volatility < 0.025:
                score += 3.5
            else:
                score += 1.5

        return min(score, 50)

    def _calculate_technical_score(self, df: pd.DataFrame) -> float:
        """
        计算技术面得分（30分）

        指标说明：
        1. 趋势强度（10分）: MA排列、均线斜率、多周期趋势
        2. 动量指标（8分）: RSI, MACD, 价格相对强度
        3. 成交量确认（6分）: 量价配合、突破确认
        4. 资金流向（6分）: 主力资金、北向资金

        数据来源：
        - Baostock: OHLCV, amount, turn
        - Tushare: moneyflow (主力资金), hk_hold (北向资金)
        """
        score = 0.0

        # 1. 趋势强度（10分）
        trend_score = 0

        # MA排列
        ma_score = 0
        if df["ma5"].iloc[-1] > df["ma20"].iloc[-1]:
            ma_score += 2
        if df["ma20"].iloc[-1] > df["ma60"].iloc[-1]:
            ma_score += 3
        if df["ma60"].iloc[-1] > df["ma120"].iloc[-1]:
            ma_score += 3
        if df["close"].iloc[-1] > df["ma250"].iloc[-1]:
            ma_score += 2

        # 均线斜率
        slope_score = 0
        if df["ma20_slope"].iloc[-1] > 0:
            slope_score += 1.5
        if df["ma60_slope"].iloc[-1] > 0:
            slope_score += 1.5

        trend_score = (ma_score + slope_score) / 10 * 10
        score += min(trend_score, 10)

        # 2. 动量指标（8分）
        momentum_score = 0

        # RSI
        rsi = df["rsi14"].iloc[-1]
        if 50 < rsi < 70:
            momentum_score += 3
        elif 30 < rsi < 50:
            momentum_score += 2
        elif rsi > 70:
            momentum_score += 1

        # MACD
        macd = df["macd"].iloc[-1]
        macd_signal = df["macd_signal"].iloc[-1]
        df["macd_hist"].iloc[-1]

        if macd > macd_signal and macd > 0:
            momentum_score += 3
        elif macd > macd_signal:
            momentum_score += 1.5

        # 价格相对强度（相对市场）
        momentum_20d = df["momentum_20d"].iloc[-1]
        if momentum_20d > 0.05:
            momentum_score += 2
        elif momentum_20d > 0:
            momentum_score += 1

        score += min(momentum_score, 8)

        # 3. 成交量确认（6分）
        volume_score = 0

        volume_ratio = df["volume_ratio"].iloc[-1]
        price_change = df["pctChg"].iloc[-1]

        # 上涨放量
        if price_change > 0 and volume_ratio > 1.2:
            volume_score += 3
        elif price_change > 0 and volume_ratio > 1.0:
            volume_score += 2
        elif price_change > 0:
            volume_score += 1

        # 下跌缩量
        elif price_change < 0 and volume_ratio < 0.8:
            volume_score += 2

        score += min(volume_score, 6)

        # 4. 资金流向（6分）
        # TODO: 集成Tushare的moneyflow接口获取真实资金流向
        fund_flow_score = 0

        # 基于价格和成交量模拟
        if price_change > 0 and volume_ratio > 1.5:
            fund_flow_score += 4
        elif price_change > 0 and volume_ratio > 1.2:
            fund_flow_score += 3
        elif price_change > 0:
            fund_flow_score += 2

        score += min(fund_flow_score, 6)

        return min(score, 30)

    def _calculate_risk_score(self, df: pd.DataFrame) -> float:
        """
        计算风险与质量得分（20分）

        指标说明：
        1. 波动性风险（6分）: 年化波动率、ATR
        2. 流动性风险（4分）: 日均成交额、换手率稳定性
        3. 公司治理（5分）: 股权结构、信息披露、管理层稳定性
        4. 行业地位（5分）: 市场份额、行业排名、竞争壁垒

        数据来源：
        - Baostock: OHLCV, turn
        - Tushare: shareholder (股权结构), disclosure (信息披露)
        """
        score = 0.0

        # 1. 波动性风险（6分）- 波动率越低得分越高
        volatility = df["volatility"].iloc[-1]

        if volatility < 0.015:
            score += 6
        elif volatility < 0.020:
            score += 5
        elif volatility < 0.025:
            score += 4
        elif volatility < 0.035:
            score += 3
        elif volatility < 0.050:
            score += 2
        else:
            score += 1

        # 2. 流动性风险（4分）- 基于成交量
        df["volume"].iloc[-20:].mean()
        avg_amount = df["amount"].iloc[-20:].mean() if "amount" in df.columns else 0

        # 估算成交额
        if avg_amount > 1000000000:  # 10亿
            score += 4
        elif avg_amount > 500000000:  # 5亿
            score += 3
        elif avg_amount > 100000000:  # 1亿
            score += 2
        elif avg_amount > 50000000:  # 5000万
            score += 1

        # 换手率稳定性
        if "turnover_rate" in df.columns:
            turnover_std = df["turnover_rate"].iloc[-20:].std()
            avg_turnover = df["turnover_rate"].iloc[-20:].mean()
            if avg_turnover > 0:
                turnover_ratio = turnover_std / avg_turnover
                if turnover_ratio < 0.3:
                    score += 2
                elif turnover_ratio < 0.5:
                    score += 1

        # 3. 趋势稳定性（5分）
        trend_stability = 0

        # 检查均线排列的稳定性
        ma_alignment = 0
        if df["ma5"].iloc[-1] > df["ma20"].iloc[-1]:
            ma_alignment += 1
        if df["ma20"].iloc[-1] > df["ma60"].iloc[-1]:
            ma_alignment += 1
        if df["ma60"].iloc[-1] > df["ma120"].iloc[-1]:
            ma_alignment += 1

        trend_stability = ma_alignment / 3 * 5
        score += min(trend_stability, 5)

        # 4. 价格位置（5分）- 在120日高点附近要谨慎
        price_position = (df["close"].iloc[-1] - df["low_120"].iloc[-1]) / (
            df["high_120"].iloc[-1] - df["low_120"].iloc[-1]
        )

        if 0.3 < price_position < 0.7:
            score += 5
        elif 0.2 < price_position < 0.8:
            score += 4
        elif 0.1 < price_position < 0.9:
            score += 3
        else:
            score += 1

        return min(score, 20)

    def _get_fundamental_details(self, fundamentals: Optional[Dict]) -> Dict:
        """获取基本面详细信息"""
        if fundamentals:
            return {
                "roe": fundamentals.get("roe", 0),
                "roic": fundamentals.get("roic", 0),
                "revenue_growth": fundamentals.get("revenue_growth", 0),
                "eps_growth": fundamentals.get("eps_growth", 0),
                "pe": fundamentals.get("pe", 0),
                "pb": fundamentals.get("pb", 0),
                "debt_ratio": fundamentals.get("debt_ratio", 0),
                "cash_flow_ratio": fundamentals.get("cash_flow_ratio", 0),
                "dividend_ratio": fundamentals.get("dividend_ratio", 0),
            }
        else:
            return {
                "roe": 0,
                "roic": 0,
                "revenue_growth": 0,
                "eps_growth": 0,
                "pe": 0,
                "pb": 0,
                "debt_ratio": 0,
                "cash_flow_ratio": 0,
                "dividend_ratio": 0,
            }

    def _get_technical_details(self, df: pd.DataFrame) -> Dict:
        """获取技术面详细信息"""
        return {
            "ma5": df["ma5"].iloc[-1],
            "ma20": df["ma20"].iloc[-1],
            "ma60": df["ma60"].iloc[-1],
            "ma120": df["ma120"].iloc[-1],
            "ma250": df["ma250"].iloc[-1],
            "rsi14": df["rsi14"].iloc[-1],
            "macd": df["macd"].iloc[-1],
            "macd_signal": df["macd_signal"].iloc[-1],
            "volume_ratio": df["volume_ratio"].iloc[-1],
            "momentum_20d": df["momentum_20d"].iloc[-1],
        }

    def _get_risk_details(self, df: pd.DataFrame) -> Dict:
        """获取风险详细信息"""
        return {
            "volatility": df["volatility"].iloc[-1],
            "atr_14": df["atr_14"].iloc[-1],
            "atr_20": df["atr_20"].iloc[-1],
            "volume_ma20": df["volume_ma20"].iloc[-1],
            "high_120": df["high_120"].iloc[-1],
            "low_120": df["low_120"].iloc[-1],
        }

    def _get_key_metrics(self, df: pd.DataFrame) -> Dict:
        """获取关键指标"""
        return {
            "close": df["close"].iloc[-1],
            "pct_change": df["pctChg"].iloc[-1],
            "volume": df["volume"].iloc[-1],
            "amount": df["amount"].iloc[-1] if "amount" in df.columns else 0,
            "ma20": df["ma20"].iloc[-1],
            "ma60": df["ma60"].iloc[-1],
        }

    def _get_rating(self, score: float) -> Dict:
        """获取评级"""
        if score >= 90:
            return {"grade": "A+", "recommendation": "强烈买入", "expected_return": "高"}
        elif score >= 80:
            return {"grade": "A", "recommendation": "买入", "expected_return": "中高"}
        elif score >= 70:
            return {"grade": "B+", "recommendation": "增持", "expected_return": "中等"}
        elif score >= 60:
            return {"grade": "B", "recommendation": "持有", "expected_return": "中低"}
        elif score >= 50:
            return {"grade": "C", "recommendation": "中性", "expected_return": "低"}
        elif score >= 40:
            return {"grade": "D", "recommendation": "减持", "expected_return": "负"}
        else:
            return {"grade": "F", "recommendation": "卖出", "expected_return": "高负"}


class TradingDecisionEngine:
    """交易决策引擎"""

    def __init__(self):
        self.scoring_system = StockScoringSystem()

    def make_decision(
        self, df: pd.DataFrame, stock_code: str = None, position_info: Dict = None, account_value: float = 1000000
    ) -> Dict:
        """
        生成交易决策

        Args:
            df: 股票数据
            stock_code: 股票代码
            position_info: 当前持仓信息
            account_value: 账户总资产

        Returns:
            交易决策
        """
        # 计算评分
        score_result = self.scoring_system.calculate_score(df, stock_code)

        # 判断趋势
        trend = self._get_trend(df)

        # 检测信号
        buy_signals = self._detect_buy_signals(df)
        sell_signals = self._detect_sell_signals(df)

        # 计算支撑阻力位
        support_levels = self._calculate_support_levels(df)
        resistance_levels = self._calculate_resistance_levels(df)

        # 生成决策
        if position_info and position_info.get("shares", 0) > 0:
            decision = self._decide_for_holding(
                df, buy_signals, sell_signals, trend, position_info, support_levels, resistance_levels
            )
        else:
            decision = self._decide_for_buying(
                df, buy_signals, sell_signals, trend, score_result, account_value, support_levels, resistance_levels
            )

        # 添加评分和趋势信息
        decision["score_result"] = score_result
        decision["trend"] = trend
        decision["buy_signals"] = buy_signals
        decision["sell_signals"] = sell_signals
        decision["support_levels"] = support_levels
        decision["resistance_levels"] = resistance_levels

        logger.info(f"股票{stock_code}交易决策: {decision['action']}, 置信度: {decision.get('confidence', 0):.2f}")

        return decision

    def _get_trend(self, df: pd.DataFrame) -> str:
        """
        判断趋势

        多时间框架趋势分析：
        - 日线趋势: MA5/MA20/MA60排列
        - 价格位置: 相对于120日高低点

        数据来源: Baostock (OHLCV)
        """
        ma20 = df["ma20"].iloc[-1]
        ma60 = df["ma60"].iloc[-1]
        ma120 = df["ma120"].iloc[-1]
        close = df["close"].iloc[-1]

        # 趋势强度
        trend_score = 0
        if close > ma20 > ma60 > ma120:
            trend_score += 3
        elif close > ma20 > ma60:
            trend_score += 2
        elif close > ma20:
            trend_score += 1
        elif close < ma20 < ma60 < ma120:
            trend_score -= 3
        elif close < ma20 < ma60:
            trend_score -= 2
        elif close < ma20:
            trend_score -= 1

        if trend_score >= 2:
            return "上升"
        elif trend_score <= -2:
            return "下降"
        else:
            return "震荡"

    def _detect_buy_signals(self, df: pd.DataFrame) -> List[Dict]:
        """
        检测买入信号

        信号类型：
        1. 均线金叉 + 放量
        2. 突破关键阻力位
        3. RSI超卖反弹
        4. MACD底背离
        5. 布林带下轨反弹

        数据来源: Baostock (OHLCV)
        """
        signals = []

        # 信号1: 均线金叉 + 放量
        if (
            len(df) >= 3
            and df["ma5"].iloc[-1] > df["ma20"].iloc[-1]
            and df["ma5"].iloc[-2] <= df["ma20"].iloc[-2]
            and df["volume_ratio"].iloc[-1] > 1.2
        ):
            signals.append({"type": "ma_crossover", "strength": "strong", "confidence": 0.8})

        # 信号2: RSI超卖反弹
        rsi = df["rsi14"].iloc[-1]
        if len(df) >= 3 and rsi < 30 and df["rsi14"].iloc[-2] < 30 and rsi > df["rsi14"].iloc[-2]:
            signals.append({"type": "rsi_oversold", "confidence": 0.6})

        # 信号3: 布林带下轨反弹
        bb_position = df["bb_position"].iloc[-1]
        if len(df) >= 3 and bb_position < 0.1 and df["close"].iloc[-1] > df["close"].iloc[-2]:
            signals.append({"type": "bollinger_bounce", "confidence": 0.65})

        # 信号4: 上升突破
        if len(df) >= 6 and df["close"].iloc[-1] > df["high"].iloc[-5:].max() and df["volume_ratio"].iloc[-1] > 1.3:
            signals.append({"type": "breakout", "confidence": 0.7})

        # 信号5: MACD金叉
        macd = df["macd"].iloc[-1]
        macd_signal = df["macd_signal"].iloc[-1]
        if len(df) >= 3 and macd > macd_signal and df["macd"].iloc[-2] <= df["macd_signal"].iloc[-2]:
            signals.append({"type": "macd_golden_cross", "confidence": 0.75})

        return signals

    def _detect_sell_signals(self, df: pd.DataFrame) -> List[Dict]:
        """
        检测卖出信号

        信号类型：
        1. 均线死叉
        2. RSI超买
        3. 跌破支撑位
        4. 放量滞涨
        5. MACD顶背离

        数据来源: Baostock (OHLCV)
        """
        signals = []

        # 信号1: 均线死叉
        if len(df) >= 3 and df["ma5"].iloc[-1] < df["ma20"].iloc[-1] and df["ma5"].iloc[-2] >= df["ma20"].iloc[-2]:
            signals.append({"type": "ma_death_cross", "strength": "strong", "confidence": 0.8})

        # 信号2: RSI超买
        rsi = df["rsi14"].iloc[-1]
        if rsi > 70:
            signals.append({"type": "rsi_overbought", "confidence": 0.7})

        # 信号3: 放量滞涨
        if len(df) >= 3 and df["pctChg"].iloc[-1] < 0.01 and df["volume_ratio"].iloc[-1] > 2.0:
            signals.append({"type": "volume_spike_no_gain", "confidence": 0.65})

        # 信号4: 下跌突破
        if len(df) >= 6 and df["close"].iloc[-1] < df["low"].iloc[-5:].min() and df["volume_ratio"].iloc[-1] > 1.3:
            signals.append({"type": "breakdown", "confidence": 0.75})

        # 信号5: MACD死叉
        macd = df["macd"].iloc[-1]
        macd_signal = df["macd_signal"].iloc[-1]
        if len(df) >= 3 and macd < macd_signal and df["macd"].iloc[-2] >= df["macd_signal"].iloc[-2]:
            signals.append({"type": "macd_death_cross", "confidence": 0.75})

        return signals

    def _calculate_support_levels(self, df: pd.DataFrame) -> Dict:
        """计算支撑位"""
        df["close"].iloc[-1]
        low_20 = df["low"].iloc[-20:].min()
        low_60 = df["low"].iloc[-60:].min()

        return {"primary": low_20, "secondary": low_60, "ma20": df["ma20"].iloc[-1], "ma60": df["ma60"].iloc[-1]}

    def _calculate_resistance_levels(self, df: pd.DataFrame) -> Dict:
        """计算阻力位"""
        high_20 = df["high"].iloc[-20:].max()
        high_60 = df["high"].iloc[-60:].max()

        return {"primary": high_20, "secondary": high_60, "ma20": df["ma20"].iloc[-1], "ma60": df["ma60"].iloc[-1]}

    def _decide_for_buying(
        self,
        df: pd.DataFrame,
        buy_signals: List[Dict],
        sell_signals: List[Dict],
        trend: str,
        score_result: Dict,
        account_value: float,
        support_levels: Dict,
        resistance_levels: Dict,
    ) -> Dict:
        """
        空仓时的买入决策

        买入条件：
        1. 趋势向好（上升或震荡）
        2. 有买入信号（至少1个）
        3. 评分达标（>=60分）
        4. 成交量确认
        5. 风险收益比合理（>=2:1）
        """
        # 计算买入分数
        buy_score = 0.0

        # 评分权重
        if score_result["total_score"] >= 70:
            buy_score += 0.3
        elif score_result["total_score"] >= 60:
            buy_score += 0.2

        # 趋势权重
        if trend == "上升":
            buy_score += 0.3
        elif trend == "震荡":
            buy_score += 0.1

        # 买入信号权重
        strong_buy_signals = [s for s in buy_signals if s.get("strength") == "strong"]
        if len(strong_buy_signals) > 0:
            buy_score += 0.3
        elif len(buy_signals) >= 2:
            buy_score += 0.2
        elif len(buy_signals) >= 1:
            buy_score += 0.1

        # 卖出信号扣分
        strong_sell_signals = [s for s in sell_signals if s.get("strength") == "strong"]
        if len(strong_sell_signals) > 0:
            buy_score -= 0.3

        if buy_score >= 0.7 and len(buy_signals) > 0:
            # 计算建议仓位
            position_size = self._calculate_position_size(score_result["total_score"], account_value)

            # 计算止损止盈
            stop_loss = self._calculate_stop_loss(df, support_levels)
            take_profit = self._calculate_take_profit(df, stop_loss, resistance_levels)

            return {
                "action": "BUY",
                "confidence": buy_score,
                "position_size": position_size,
                "position_value": account_value * position_size,
                "buy_price": df["close"].iloc[-1],
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "signals": buy_signals,
                "reason": f"满足买入条件，得分: {buy_score:.2f}",
            }
        else:
            return {
                "action": "WAIT",
                "confidence": buy_score,
                "reason": f"买入条件不足，得分: {buy_score:.2f}",
                "missing_conditions": self._get_missing_buy_conditions(buy_signals, trend, score_result),
            }

    def _decide_for_holding(
        self,
        df: pd.DataFrame,
        buy_signals: List[Dict],
        sell_signals: List[Dict],
        trend: str,
        position_info: Dict,
        support_levels: Dict,
        resistance_levels: Dict,
    ) -> Dict:
        """
        持仓时的持有/卖出决策

        卖出条件：
        1. 触发止损
        2. 强卖出信号
        3. 趋势反转
        4. 达到止盈目标
        5. 时间止损（60天）

        持有条件：
        1. 趋势仍向上
        2. 无卖出信号
        3. 价格在MA20上方
        4. 成交量正常
        """
        # 卖出紧急程度
        sell_urgency = 0.0

        # 检查止损
        current_price = df["close"].iloc[-1]
        entry_price = position_info.get("entry_price", current_price)
        stop_loss = position_info.get("stop_loss", entry_price * 0.9)

        if current_price <= stop_loss:
            sell_urgency += 1.0

        # 强卖出信号
        strong_sell_signals = [s for s in sell_signals if s.get("strength") == "strong"]
        if len(strong_sell_signals) > 0:
            sell_urgency += 0.8

        # 趋势反转
        if trend == "下降" and position_info.get("entry_trend") == "上升":
            sell_urgency += 0.6

        # 达到止盈
        take_profit = position_info.get("take_profit", {})
        if take_profit and current_price >= take_profit.get("tp1", float("inf")):
            sell_urgency += 0.5

        # 持有条件
        hold_score = 0.0

        if trend == "上升":
            hold_score += 0.3
        elif trend == "震荡":
            hold_score += 0.2

        if current_price > df["ma20"].iloc[-1]:
            hold_score += 0.2

        if df["volume_ratio"].iloc[-1] > 0.8:
            hold_score += 0.1

        if len(sell_signals) == 0:
            hold_score += 0.3

        # 决策
        if sell_urgency >= 1.0:
            return {
                "action": "SELL_ALL",
                "urgency": "high",
                "reason": "触发强制卖出条件",
                "current_pnl": (current_price - entry_price) / entry_price * 100,
                "sell_price": current_price,
            }
        elif sell_urgency >= 0.7:
            return {
                "action": "SELL_PARTIAL",
                "percentage": 0.5,
                "reason": "出现强烈卖出信号",
                "sell_signals": sell_signals,
                "sell_price": current_price,
            }
        elif hold_score >= 0.5:
            return {
                "action": "HOLD",
                "confidence": hold_score,
                "reason": "持有条件满足",
                "adjust_stop_loss": current_price * 0.95,
                "current_pnl": (current_price - entry_price) / entry_price * 100,
            }
        else:
            return {
                "action": "REDUCE",
                "percentage": 0.3,
                "reason": "部分持有条件不满足",
                "hold_score": hold_score,
                "sell_price": current_price,
            }

    def _calculate_position_size(self, score: float, account_value: float) -> float:
        """
        计算建议仓位大小

        基于评分计算：
        - A+ (90+): 30%
        - A (80-89): 25%
        - B+ (70-79): 20%
        - B (60-69): 15%
        - 低于60分: 不买入
        """
        if score >= 90:
            return 0.3  # 30%
        elif score >= 80:
            return 0.25  # 25%
        elif score >= 70:
            return 0.2  # 20%
        elif score >= 60:
            return 0.15  # 15%
        else:
            return 0.0

    def _calculate_stop_loss(self, df: pd.DataFrame, support_levels: Dict) -> float:
        """
        计算止损价

        止损策略：
        1. ATR止损（2倍ATR）
        2. 支撑位止损（下方2%）
        3. 百分比止损（8%）

        选择最严格的止损
        """
        close = df["close"].iloc[-1]
        atr = df["atr_14"].iloc[-1]

        # ATR止损（2倍ATR）
        atr_stop = close - atr * 2

        # 支撑位止损
        support_stop = support_levels["primary"] * 0.98

        # 百分比止损（8%）
        percent_stop = close * 0.92

        return max(atr_stop, support_stop, percent_stop)

    def _calculate_take_profit(self, df: pd.DataFrame, stop_loss: float, resistance_levels: Dict) -> Dict:
        """
        计算止盈位

        止盈策略：
        1. 盈亏比目标（至少2:1）
        2. 第一目标位（基于阻力位）
        3. 分批止盈（50%/30%/20%）
        """
        close = df["close"].iloc[-1]
        df["atr_14"].iloc[-1]

        risk = close - stop_loss

        # 第一目标位（2:1盈亏比）
        tp1 = close + risk * 2

        # 考虑阻力位
        if resistance_levels["primary"] < tp1:
            tp1 = resistance_levels["primary"]

        # 第二目标位
        tp2 = tp1 * 1.2

        # 第三目标位
        tp3 = tp1 * 1.4

        return {"tp1": tp1, "tp2": tp2, "tp3": tp3, "trailing_stop": tp1 * 0.95}

    def _get_missing_buy_conditions(self, buy_signals: List[Dict], trend: str, score_result: Dict) -> List[str]:
        """获取缺失的买入条件"""
        missing = []

        if len(buy_signals) == 0:
            missing.append("买入信号")

        if score_result["total_score"] < 70:
            missing.append("高分值（≥70分）")

        if trend != "上升":
            missing.append("上升趋势")

        return missing


if __name__ == "__main__":
    start_time = datetime.now().timestamp()
    failure_reason = None
    setup_logging("stock_selection")

    print("=" * 70)
    print("股票评分和交易决策系统测试")
    print("=" * 70)

    try:
        df = pd.read_parquet("/Users/dongxg/SourceCode/deep_final_kp/data/processed/stocks_2025.parquet")
        df_stock = df[df["code"] == "sh.600000"].copy().sort_values("date").reset_index(drop=True)

        print("\n测试股票: sh.600000 (浦发银行)")
        print(f"数据范围: {df_stock['date'].min()} ~ {df_stock['date'].max()}")
        print(f"数据点数: {len(df_stock)}")

        scoring_system = StockScoringSystem()

        print("\n计算股票评分...")
        score_result = scoring_system.calculate_score(df_stock, "sh.600000")

        print(f"\n{'='*70}")
        print("评分结果")
        print(f"{'='*70}")
        print(f"总分: {score_result['total_score']:.2f}")
        print(f"评级: {score_result['rating']}")
        print(f"建议: {score_result['recommendation']}")
        print(f"预期收益: {score_result['expected_return']}")
        print("\n分项得分:")
        print(f"  基本面: {score_result['fundamental_score']:.2f}/50")
        print(f"  技术面: {score_result['technical_score']:.2f}/30")
        print(f"  风险质量: {score_result['risk_score']:.2f}/20")

        decision_engine = TradingDecisionEngine()

        print("\n生成交易决策...")
        decision = decision_engine.make_decision(df_stock, "sh.600000")

        print(f"\n{'='*70}")
        print("交易决策")
        print(f"{'='*70}")
        print(f"操作: {decision['action']}")
        print(f"置信度: {decision.get('confidence', 0):.2f}")

        if decision["action"] == "BUY":
            print(f"买入价格: {decision['buy_price']:.2f}")
            print(f"建议仓位: {decision['position_size']*100:.0f}%")
            print(f"建议金额: {decision['position_value']:,.0f}元")
            print(f"止损价: {decision['stop_loss']:.2f}")
            print("止盈价:")
            print(f"  TP1: {decision['take_profit']['tp1']:.2f} (50%)")
            print(f"  TP2: {decision['take_profit']['tp2']:.2f} (30%)")
            print(f"  TP3: {decision['take_profit']['tp3']:.2f} (20%)")
            print(f"  移动止损: {decision['take_profit']['trailing_stop']:.2f}")
            print(f"买入信号: {[s['type'] for s in decision['buy_signals']]}")
            print(f"理由: {decision['reason']}")

        elif decision["action"] == "HOLD":
            print(f"理由: {decision['reason']}")
            print(f"调整止损: {decision.get('adjust_stop_loss', 'N/A'):.2f}")

        elif decision["action"] == "WAIT":
            print(f"理由: {decision['reason']}")
            print(f"缺失条件: {decision.get('missing_conditions', [])}")

        print(f"\n趋势状态: {decision['trend']}")
        print(f"买入信号数: {len(decision['buy_signals'])}")
        print(f"卖出信号数: {len(decision['sell_signals'])}")

        print("\n" + "=" * 70)
        print("测试完成！")
        print("=" * 70)

    except Exception as exc:
        failure_reason = str(exc)
        print(f"测试失败: {exc}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        elapsed_s = datetime.now().timestamp() - start_time
        logger.info(
            format_task_summary(
                "stock_scoring_system_demo",
                window=None,
                elapsed_s=elapsed_s,
                error=failure_reason,
            )
        )
