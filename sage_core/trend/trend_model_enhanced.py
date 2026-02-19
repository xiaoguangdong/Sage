"""
趋势状态模型（增强版 - 12状态分类）

基于 docs/market_tendcy_qa.md 的扩展市场状态分类体系：
- 牛市阶段：INITIATING_BULL, CONFIRMED_BULL, LATE_BULL, BULL_CORRECTION
- 熊市阶段：INITIATING_BEAR, CONFIRMED_BEAR, LATE_BEAR, BEAR_RALLY
- 震荡阶段：ASCENDING_CONSOLIDATION, DESCENDING_CONSOLIDATION, RANGE_BOUND, EXPANDING_VOLATILITY
- 特殊状态：EXTREME_OVERSOLD, EXTREME_OVERBOUGHT, TREND_EXHAUSTION
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TrendModelEnhanced:
    """增强版趋势状态模型（12状态分类）"""

    # 状态定义
    STATES = {
        # 牛市阶段
        "INITIATING_BULL": {"id": 1, "name": "初始牛市", "category": "bull"},
        "CONFIRMED_BULL": {"id": 2, "name": "确认牛市", "category": "bull"},
        "LATE_BULL": {"id": 3, "name": "晚期牛市", "category": "bull"},
        "BULL_CORRECTION": {"id": 4, "name": "牛市调整", "category": "bull"},
        # 熊市阶段
        "INITIATING_BEAR": {"id": -1, "name": "初始熊市", "category": "bear"},
        "CONFIRMED_BEAR": {"id": -2, "name": "主跌熊市", "category": "bear"},
        "LATE_BEAR": {"id": -3, "name": "晚期熊市", "category": "bear"},
        "BEAR_RALLY": {"id": -4, "name": "熊市反弹", "category": "bear"},
        # 震荡阶段
        "ASCENDING_CONSOLIDATION": {"id": 10, "name": "上升中继", "category": "consolidation"},
        "DESCENDING_CONSOLIDATION": {"id": 20, "name": "下降中继", "category": "consolidation"},
        "RANGE_BOUND": {"id": 30, "name": "箱体震荡", "category": "consolidation"},
        "EXPANDING_VOLATILITY": {"id": 40, "name": "扩张三角", "category": "consolidation"},
        # 特殊状态
        "EXTREME_OVERSOLD": {"id": 100, "name": "极度超卖", "category": "special"},
        "EXTREME_OVERBOUGHT": {"id": 200, "name": "极度超买", "category": "special"},
        "TREND_EXHAUSTION": {"id": 300, "name": "趋势衰竭", "category": "special"},
    }

    def __init__(
        self,
        ma_short: int = 20,
        ma_medium: int = 60,
        ma_long: int = 120,
        rsi_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
    ):
        """
        初始化增强版趋势模型

        Args:
            ma_short: 短期均线周期
            ma_medium: 中期均线周期
            ma_long: 长期均线周期
            rsi_period: RSI周期
            bb_period: 布林带周期
            bb_std: 布林带标准差倍数
        """
        self.ma_short = ma_short
        self.ma_medium = ma_medium
        self.ma_long = ma_long
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.bb_std = bb_std

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有技术指标"""
        df = df.copy()

        # 移动平均线
        df[f"ma_{self.ma_short}"] = df["close"].rolling(self.ma_short).mean()
        df[f"ma_{self.ma_medium}"] = df["close"].rolling(self.ma_medium).mean()
        df[f"ma_{self.ma_long}"] = df["close"].rolling(self.ma_long).mean()

        # 均线斜率
        df[f"ma_{self.ma_short}_slope"] = df[f"ma_{self.ma_short}"].diff(5)
        df[f"ma_{self.ma_medium}_slope"] = df[f"ma_{self.ma_medium}"].diff(5)

        # MACD
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # 布林带
        df["bb_middle"] = df["close"].rolling(self.bb_period).mean()
        df["bb_std"] = df["close"].rolling(self.bb_period).std()
        df["bb_upper"] = df["bb_middle"] + self.bb_std * df["bb_std"]
        df["bb_lower"] = df["bb_middle"] - self.bb_std * df["bb_std"]
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # 波动率
        df["volatility"] = df["close"].pct_change().rolling(20).std()
        df["volatility_ma"] = df["volatility"].rolling(60).median()

        # 动量
        df["momentum_4w"] = df["close"].pct_change(4)
        df["momentum_12w"] = df["close"].pct_change(12)

        # 乖离率
        df["bias"] = (
            (df["close"] - df["close"].rolling(self.ma_medium).mean())
            / df["close"].rolling(self.ma_medium).mean()
            * 100
        )

        return df

    def _detect_trend(self, df: pd.DataFrame, idx: int) -> str:
        """检测趋势方向"""
        row = df.iloc[idx]
        ma20 = row[f"ma_{self.ma_short}"]
        ma60 = row[f"ma_{self.ma_medium}"]
        ma120 = row[f"ma_{self.ma_long}"]
        close = row["close"]

        # 判断均线排列
        if ma20 > ma60 > ma120 and close > ma120:
            return "bull"
        elif ma20 < ma60 < ma120 and close < ma120:
            return "bear"
        else:
            return "neutral"

    def _detect_bull_state(self, df: pd.DataFrame, idx: int) -> Tuple[str, float]:
        """检测牛市阶段"""
        row = df.iloc[idx]

        ma20_slope = row[f"ma_{self.ma_short}_slope"]
        ma60_slope = row[f"ma_{self.ma_medium}_slope"]
        rsi = row["rsi"]
        macd = row["macd"]
        macd_signal = row["macd_signal"]
        bb_position = row["bb_position"]
        momentum_4w = row["momentum_4w"]
        volatility = row["volatility"]
        vol_ma = row["volatility_ma"]
        bias = row["bias"]

        # 1. 检测牛市调整
        if idx > 0:
            prev_state = df.iloc[idx - 1]["state"] if "state" in df.columns else None
            if prev_state in ["INITIATING_BULL", "CONFIRMED_BULL"]:
                # 在牛市中出现回调
                if momentum_4w < -0.05 and bias > -10:
                    return "BULL_CORRECTION", 0.8

        # 2. 检测晚期牛市
        if rsi > 75 and volatility > vol_ma * 1.5 and bb_position > 0.8 and macd > 0 and ma20_slope > 0:
            return "LATE_BULL", 0.75

        # 3. 检测确认牛市
        if (
            ma20_slope > 0
            and ma60_slope > 0
            and rsi > 50
            and rsi < 70
            and momentum_4w > 0
            and volatility < vol_ma * 1.2
        ):
            return "CONFIRMED_BULL", 0.8

        # 4. 检测初始牛市
        if ma20_slope > 0 and momentum_4w > 0 and rsi > 45 and rsi < 60 and macd > macd_signal:
            return "INITIATING_BULL", 0.7

        return "CONFIRMED_BULL", 0.6

    def _detect_bear_state(self, df: pd.DataFrame, idx: int) -> Tuple[str, float]:
        """检测熊市阶段"""
        row = df.iloc[idx]

        ma20_slope = row[f"ma_{self.ma_short}_slope"]
        ma60_slope = row[f"ma_{self.ma_medium}_slope"]
        rsi = row["rsi"]
        macd = row["macd"]
        macd_signal = row["macd_signal"]
        bb_position = row["bb_position"]
        momentum_4w = row["momentum_4w"]
        volatility = row["volatility"]
        vol_ma = row["volatility_ma"]
        bias = row["bias"]

        # 1. 检测熊市反弹
        if idx > 0:
            prev_state = df.iloc[idx - 1]["state"] if "state" in df.columns else None
            if prev_state in ["INITIATING_BEAR", "CONFIRMED_BEAR"]:
                # 在熊市中出现反弹
                if momentum_4w > 0.05 and bias < 10:
                    return "BEAR_RALLY", 0.75

        # 2. 检测晚期熊市（寻底期）
        if rsi < 30 and volatility < vol_ma * 0.8 and bb_position < 0.2 and momentum_4w > -0.1:
            return "LATE_BEAR", 0.7

        # 3. 检测主跌熊市
        if ma20_slope < 0 and ma60_slope < 0 and rsi < 50 and momentum_4w < 0 and volatility > vol_ma * 1.2:
            return "CONFIRMED_BEAR", 0.8

        # 4. 检测初始熊市
        if ma20_slope < 0 and momentum_4w < 0 and rsi < 55 and rsi > 40 and macd < macd_signal:
            return "INITIATING_BEAR", 0.7

        return "CONFIRMED_BEAR", 0.6

    def _detect_consolidation_state(self, df: pd.DataFrame, idx: int) -> Tuple[str, float]:
        """检测震荡阶段"""
        row = df.iloc[idx]

        ma20_slope = row[f"ma_{self.ma_short}_slope"]
        ma60_slope = row[f"ma_{self.ma_medium}_slope"]
        bb_width = row["bb_width"]
        volatility = row["volatility"]
        vol_ma = row.get("volatility_ma", np.nan)
        rsi = row["rsi"]
        momentum_4w = row["momentum_4w"]

        # 1. 检测扩张三角（波动加大）
        if bb_width > 0.15 and pd.notna(vol_ma) and volatility > vol_ma * 1.5:
            return "EXPANDING_VOLATILITY", 0.7

        # 2. 检测箱体震荡
        if abs(ma20_slope) < 0.01 and abs(ma60_slope) < 0.01 and bb_width < 0.10 and rsi > 40 and rsi < 60:
            return "RANGE_BOUND", 0.7

        # 3. 检测上升中继
        if ma20_slope > 0 and ma60_slope > 0 and momentum_4w > -0.02 and momentum_4w < 0.02 and bb_width < 0.08:
            return "ASCENDING_CONSOLIDATION", 0.65

        # 4. 检测下降中继
        if ma20_slope < 0 and ma60_slope < 0 and momentum_4w > -0.02 and momentum_4w < 0.02 and bb_width < 0.08:
            return "DESCENDING_CONSOLIDATION", 0.65

        return "RANGE_BOUND", 0.5

    def _detect_special_state(self, df: pd.DataFrame, idx: int) -> Optional[Tuple[str, float]]:
        """检测特殊状态"""
        row = df.iloc[idx]

        rsi = row["rsi"]
        bb_position = row["bb_position"]
        bias = row["bias"]
        macd = row["macd"]
        row["macd_signal"]
        row["momentum_4w"]

        # 极度超卖
        if rsi < 20 and bb_position < 0 and bias < -10:
            return "EXTREME_OVERSOLD", 0.9

        # 极度超买
        if rsi > 80 and bb_position > 1 and bias > 10:
            return "EXTREME_OVERBOUGHT", 0.9

        # 趋势衰竭（MACD顶底背离）
        if idx >= 5:
            recent_prices = df.iloc[idx - 5 : idx + 1]["close"].values
            recent_macd = df.iloc[idx - 5 : idx + 1]["macd"].values

            # 价格创新高但MACD走弱（顶背离）
            if recent_prices[-1] > recent_prices[0] and recent_macd[-1] < recent_macd[0] and macd > 0:
                return "TREND_EXHAUSTION", 0.8

            # 价格创新低但MACD走强（底背离）
            if recent_prices[-1] < recent_prices[0] and recent_macd[-1] > recent_macd[0] and macd < 0:
                return "TREND_EXHAUSTION", 0.8

        return None

    def predict(self, df: pd.DataFrame, idx: int = None) -> Dict:
        """
        预测市场状态

        Args:
            df: 指数数据DataFrame，必须包含'close'列
            idx: 预测的索引位置，如果为None则预测最后一个数据点

        Returns:
            包含状态和置信度的字典
        """
        if "close" not in df.columns:
            raise ValueError("DataFrame必须包含'close'列")

        if idx is None:
            idx = len(df) - 1

        # 计算指标
        df_with_indicators = self._calculate_indicators(df)

        # 检查数据是否足够
        if idx < max(self.ma_long, self.bb_period, self.rsi_period):
            logger.warning(f"数据不足，需要至少{max(self.ma_long, self.bb_period, self.rsi_period)}条数据")
            return {
                "state": "UNKNOWN",
                "state_id": 0,
                "state_name": "数据不足",
                "confidence": 0.0,
                "category": "unknown",
            }

        # 1. 优先检测特殊状态
        special_state = self._detect_special_state(df_with_indicators, idx)
        if special_state:
            state_key, confidence = special_state
            return {
                "state": state_key,
                "state_id": self.STATES[state_key]["id"],
                "state_name": self.STATES[state_key]["name"],
                "confidence": confidence,
                "category": self.STATES[state_key]["category"],
            }

        # 2. 检测趋势方向
        trend = self._detect_trend(df_with_indicators, idx)

        # 3. 根据趋势方向检测具体状态
        if trend == "bull":
            state_key, confidence = self._detect_bull_state(df_with_indicators, idx)
        elif trend == "bear":
            state_key, confidence = self._detect_bear_state(df_with_indicators, idx)
        else:
            state_key, confidence = self._detect_consolidation_state(df_with_indicators, idx)

        return {
            "state": state_key,
            "state_id": self.STATES[state_key]["id"],
            "state_name": self.STATES[state_key]["name"],
            "confidence": confidence,
            "category": self.STATES[state_key]["category"],
        }

    def predict_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        预测所有时间点的状态

        Args:
            df: 指数数据DataFrame

        Returns:
            包含状态预测的DataFrame
        """
        df_with_indicators = self._calculate_indicators(df)

        states = []
        confidences = []
        state_ids = []
        state_names = []
        categories = []

        # 从有足够数据的位置开始预测
        start_idx = max(self.ma_long, self.bb_period, self.rsi_period)

        for i in range(start_idx, len(df_with_indicators)):
            # 检测特殊状态
            special_state = self._detect_special_state(df_with_indicators, i)
            if special_state:
                state_key, confidence = special_state
            else:
                # 检测趋势方向
                trend = self._detect_trend(df_with_indicators, i)

                # 根据趋势方向检测具体状态
                if trend == "bull":
                    state_key, confidence = self._detect_bull_state(df_with_indicators, i)
                elif trend == "bear":
                    state_key, confidence = self._detect_bear_state(df_with_indicators, i)
                else:
                    state_key, confidence = self._detect_consolidation_state(df_with_indicators, i)

            states.append(state_key)
            confidences.append(confidence)
            state_ids.append(self.STATES[state_key]["id"])
            state_names.append(self.STATES[state_key]["name"])
            categories.append(self.STATES[state_key]["category"])

        # 创建结果DataFrame
        result_df = df_with_indicators.iloc[start_idx:].copy()
        result_df["state"] = states
        result_df["confidence"] = confidences
        result_df["state_id"] = state_ids
        result_df["state_name"] = state_names
        result_df["category"] = categories

        return result_df


def create_enhanced_trend_model(config: dict = None) -> TrendModelEnhanced:
    """
    创建增强版趋势模型

    Args:
        config: 配置字典

    Returns:
        增强版趋势模型实例
    """
    if config is None:
        config = {}

    return TrendModelEnhanced(
        ma_short=config.get("ma_short", 20),
        ma_medium=config.get("ma_medium", 60),
        ma_long=config.get("ma_long", 120),
        rsi_period=config.get("rsi_period", 14),
        bb_period=config.get("bb_period", 20),
        bb_std=config.get("bb_std", 2.0),
    )


if __name__ == "__main__":
    # 测试增强版趋势模型
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("测试增强版趋势模型（12状态分类）")
    print("=" * 70)

    # 加载测试数据（使用沪深300指数）
    print("\n正在加载沪深300指数数据...")

    # 从合并的数据中获取沪深300指数
    df_2025 = pd.read_parquet("/Users/dongxg/SourceCode/deep_final_kp/data/processed/stocks_2025.parquet")

    # 筛选沪深300指数（sh.000300）
    df_index = df_2025[df_2025["code"] == "sh.000300"].copy()
    df_index = df_index.sort_values("date").reset_index(drop=True)

    print(f"数据范围: {df_index['date'].min()} ~ {df_index['date'].max()}")
    print(f"数据点数: {len(df_index)}")

    # 创建模型
    model = create_enhanced_trend_model()

    # 预测所有状态
    print("\n开始预测所有时间点的市场状态...")
    result_df = model.predict_all(df_index)

    # 统计各状态出现次数
    print("\n" + "=" * 70)
    print("市场状态统计")
    print("=" * 70)
    state_counts = result_df["state_name"].value_counts()
    for state, count in state_counts.items():
        percentage = count / len(result_df) * 100
        print(f"{state}: {count}次 ({percentage:.2f}%)")

    # 按类别统计
    print("\n按类别统计:")
    category_counts = result_df["category"].value_counts()
    for category, count in category_counts.items():
        percentage = count / len(result_df) * 100
        print(f"{category}: {count}次 ({percentage:.2f}%)")

    # 显示最近10天的预测结果
    print("\n" + "=" * 70)
    print("最近10天的市场状态")
    print("=" * 70)
    recent_results = result_df[["date", "close", "state_name", "category", "confidence"]].tail(10)
    print(recent_results.to_string(index=False))

    # 计算状态转移矩阵
    print("\n" + "=" * 70)
    print("状态转移矩阵（最近10次转移）")
    print("=" * 70)
    state_sequence = result_df["state_name"].values
    transitions = {}
    for i in range(len(state_sequence) - 1):
        from_state = state_sequence[i]
        to_state = state_sequence[i + 1]
        key = f"{from_state} -> {to_state}"
        transitions[key] = transitions.get(key, 0) + 1

    # 显示最常见的转移
    sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
    for transition, count in sorted_transitions[:10]:
        print(f"{transition}: {count}次")

    print("\n测试完成！")
