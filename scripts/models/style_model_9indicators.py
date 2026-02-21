#!/usr/bin/env python3
"""
9指标风格模型（ChatGPT方案）

基于9个核心指标，识别市场风格状态
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd

from scripts.data._shared.runtime import get_data_path, get_tushare_root, log_task_summary, setup_logger

logger = setup_logger("style_model_9indicators", module="models")


class StyleModel9Indicators:
    """9指标风格模型"""

    def __init__(self):
        self.data_dir = str(get_tushare_root())
        self.output_dir = str(get_data_path("processed", "factors", ensure=True))

        # 状态定义
        self.STATES = {
            0: "ACCUMULATION",  # 积累期
            1: "HEALTHY_UP",  # 健康扩张（牛市）
            2: "DISTRIBUTION",  # 风险转移
            3: "EXIT",  # 逻辑失效
        }

        # 阈值配置
        self.thresholds = {
            "A1_coupling": 0.3,  # 换手-涨幅耦合阈值
            "A2_liquidity_dry": 0.9,  # 回调成交枯竭率阈值
            "A3_recovery_days": 7,  # 修复天数阈值
            "B1_spread": 0.02,  # 龙头-跟风背离阈值
            "B2_impact_decay": 0.5,  # 利好钝化阈值
            "B3_dispersion": 0.05,  # 板块分化阈值
            "C1_diverge": 2.0,  # 情绪-价格背离阈值
            "C2_failed_count": 3,  # 失败反弹次数阈值
            "C3_junk_ratio": 0.3,  # 垃圾股比例阈值
        }

        logger.info("9指标风格模型初始化")

    def load_data(self):
        """加载数据"""
        logger.info("=" * 70)
        logger.info("加载数据...")
        logger.info("=" * 70)

        # 加载日线数据
        logger.info("加载日线数据...")
        daily_data = []
        for year in range(2020, 2027):
            file_path = os.path.join(self.data_dir, "daily", f"daily_{year}.parquet")
            if os.path.exists(file_path):
                df = pd.read_parquet(file_path)
                daily_data.append(df)

        if daily_data:
            self.daily = pd.concat(daily_data, ignore_index=True)
            self.daily["trade_date"] = pd.to_datetime(self.daily["trade_date"])
            logger.info(f"✓ 日线数据总计: {len(self.daily)} 条记录")
        else:
            logger.error("✗ 未找到日线数据")
            return False

        # 加载daily_basic数据
        logger.info("加载daily_basic数据...")
        basic_file = os.path.join(self.data_dir, "daily_basic_all.parquet")
        if os.path.exists(basic_file):
            self.basic = pd.read_parquet(basic_file)
            self.basic["trade_date"] = pd.to_datetime(self.basic["trade_date"])
            logger.info(f"✓ Daily basic: {len(self.basic)} 条记录")
        else:
            logger.error("✗ 未找到daily_basic数据")
            return False

        # 加载指数数据
        logger.info("加载指数数据...")
        index_file = os.path.join(self.data_dir, "index", "index_ohlc_all.parquet")
        if os.path.exists(index_file):
            self.index_data = pd.read_parquet(index_file)
            self.index_data["trade_date"] = pd.to_datetime(self.index_data["date"])
            logger.info(f"✓ 指数数据: {len(self.index_data)} 条记录")
        else:
            logger.error("✗ 未找到指数数据")
            return False

        return True

    def calc_A_factors(self, date_data, window=20):
        """计算A类指标：筹码行为"""
        # A1: 换手-涨幅耦合度
        ret = date_data["pct_chg"] / 100
        turnover = date_data["turnover_rate"]

        # 计算滚动相关性
        coupling = ret.rolling(window).corr(turnover)

        # A2: 回调成交枯竭率
        drawdown_days = ret < -0.01
        volume_mean = date_data["vol"].rolling(5).mean()
        liquidity_dry = np.where(drawdown_days, date_data["vol"] / volume_mean, np.nan)

        # A3: 高位修复效率
        # 找到局部高点
        peaks = self._find_peaks(date_data["close"], distance=5)
        recovery_days = self._calc_recovery_days(date_data["close"], peaks, window)

        return {"A1_coupling": coupling, "A2_liquidity_dry": liquidity_dry, "A3_recovery_days": recovery_days}

    def calc_B_factors(self, date_data, sector_data=None):
        """计算B类指标：定价逻辑"""
        ret = date_data["pct_chg"] / 100

        # B1: 龙头-跟风背离
        # 简化：使用前10%的股票作为龙头
        top_10pct = ret.quantile(0.9)
        bottom_90pct = ret.quantile(0.1)
        spread = top_10pct - bottom_90pct

        # B2: 利好钝化（简化版：使用价格波动）
        vol = ret.rolling(20).std()
        impact_decay = vol / vol.rolling(60).mean()

        # B3: 叙事一致性（简化版：收益分散度）
        dispersion = ret.rolling(20).std()

        return {"B1_spread": spread, "B2_impact_decay": impact_decay, "B3_dispersion": dispersion}

    def calc_C_factors(self, date_data, window=20):
        """计算C类指标：群体博弈"""
        ret = date_data["pct_chg"] / 100

        # C1: 情绪-价格背离（简化版：波动率/收益）
        vol = ret.rolling(window).std()
        emotion_diverge = vol / (ret.abs().rolling(window).mean() + 0.001)

        # C2: 失败反弹次数
        failed_rebounds = self._count_failed_rebounds(date_data["close"], window)

        # C3: 非理性扩散（简化版：小市值股票表现）
        # 这里简化为：收益分布的右偏程度
        skewness = ret.rolling(window).skew()

        return {"C1_diverge": emotion_diverge, "C2_failed_count": failed_rebounds, "C3_junk_ratio": skewness}

    def _find_peaks(self, prices, distance=5):
        """找到局部高点"""
        peaks = []
        for i in range(distance, len(prices) - distance):
            if prices[i] == max(prices[i - distance : i + distance + 1]):
                peaks.append(i)
        return peaks

    def _calc_recovery_days(self, prices, peaks, window):
        """计算修复天数"""
        recovery_days = pd.Series(np.nan, index=prices.index)

        for peak_idx in peaks:
            peak_price = prices[peak_idx]
            # 向后查找恢复到峰值的时间
            for i in range(peak_idx + 1, min(peak_idx + window, len(prices))):
                if prices[i] >= peak_price:
                    recovery_days.iloc[peak_idx] = i - peak_idx
                    break

        return recovery_days

    def _count_failed_rebounds(self, prices, window=20):
        """计算失败反弹次数"""
        failed_count = pd.Series(0, index=prices.index)

        for i in range(window, len(prices)):
            # 检查过去window天内的反弹是否失败
            local_prices = prices[i - window : i]
            if len(local_prices) < 5:
                continue

            # 找到局部低点
            local_min_idx = local_prices.idxmin()
            local_min_price = local_prices[local_min_idx]

            # 检查反弹后是否再次跌破
            if local_min_idx < i - 5:
                rebound_high = local_prices[local_min_idx:i].max()
                if rebound_high < local_min_price * 1.02:  # 反弹幅度小于2%
                    failed_count.iloc[i] += 1

        return failed_count

    def zscore(self, x, window=60):
        """Z-score标准化"""
        return (x - x.rolling(window).mean()) / (x.rolling(window).std() + 1e-8)

    def market_state(self, factors):
        """判定市场状态"""
        A = factors["A"]
        B = factors["B"]
        C = factors["C"]

        # 状态1: 健康扩张（牛市）
        if (
            A["A1_coupling"] > self.thresholds["A1_coupling"]
            and A["A2_liquidity_dry"] < self.thresholds["A2_liquidity_dry"]
            and A["A3_recovery_days"] < self.thresholds["A3_recovery_days"]
            and B["B3_dispersion"] < 0.05
        ):
            return self.STATES[1]

        # 状态3: 逻辑失效（EXIT）
        if A["A1_coupling"] < 0 or A["A2_liquidity_dry"] > 1.1:
            if C["C3_junk_ratio"] > 1.0:  # 高偏度（非理性扩散）
                return self.STATES[3]

        # 状态2: 风险转移（DISTRIBUTION）
        if A["A1_coupling"] > 0 and B["B3_dispersion"] > 0.05:
            return self.STATES[2]

        # 默认：积累期
        return self.STATES[0]

    def decision_engine(self, market_state, confidence):
        """决策引擎"""
        if market_state == "EXIT":
            return {"position": 0.0, "action": "CLEAR", "risk": "HIGH"}

        if market_state == "DISTRIBUTION":
            return {"position": 0.3, "action": "REDUCE", "risk": "MEDIUM"}

        if market_state == "HEALTHY_UP":
            return {"position": min(1.0, confidence), "action": "PARTICIPATE", "risk": "LOW"}

        # ACCUMULATION
        return {"position": 0.5, "action": "NEUTRAL", "risk": "MEDIUM"}

    def run(self):
        """执行完整的分析流程"""
        logger.info("\n" + "=" * 70)
        logger.info("开始9指标风格模型分析...")
        logger.info("=" * 70)

        if not self.load_data():
            logger.error("数据加载失败")
            return None

        # 计算市场层面的指标（使用沪深300作为代理）
        logger.info("计算市场层面指标...")
        hs300 = self.index_data[self.index_data["code"] == "000300.SH"].copy()
        hs300 = hs300.sort_values("trade_date")

        # 计算收益率
        hs300["pct_chg"] = hs300["close"].pct_change() * 100

        # 计算A类指标
        A_factors = self.calc_A_factors(hs300)

        # 计算B类指标
        B_factors = self.calc_B_factors(hs300)

        # 计算C类指标
        C_factors = self.calc_C_factors(hs300)

        # 标准化指标
        A_zscore = {k: self.zscore(v) for k, v in A_factors.items()}
        B_zscore = {k: self.zscore(v) for k, v in B_factors.items()}
        C_zscore = {k: self.zscore(v) for k, v in C_factors.items()}

        # 构建结果DataFrame
        results = pd.DataFrame({"trade_date": hs300["trade_date"], **A_zscore, **B_zscore, **C_zscore})

        # 为每个日期判定市场状态
        logger.info("判定市场状态...")
        states = []
        decisions = []
        confidences = []

        for i in range(len(results)):
            factors = {
                "A": {k: v.iloc[i] for k, v in A_factors.items()},
                "B": {k: v.iloc[i] for k, v in B_factors.items()},
                "C": {k: v.iloc[i] for k, v in C_factors.items()},
            }

            state = self.market_state(factors)
            confidence = min(1.0, max(0.0, 0.5 + np.random.normal(0, 0.1)))
            decision = self.decision_engine(state, confidence)

            states.append(state)
            decisions.append(decision["action"])
            confidences.append(decision["position"])

        results["market_state"] = states
        results["action"] = decisions
        results["position"] = confidences

        # 保存结果
        output_file = os.path.join(self.output_dir, "style_model_9indicators_results.parquet")
        results.to_parquet(output_file)
        logger.info(f"✓ 结果已保存: {output_file}")

        # 输出统计摘要
        logger.info("\n" + "=" * 70)
        logger.info("状态分布统计：")
        logger.info("=" * 70)
        state_counts = results["market_state"].value_counts()
        for state, count in state_counts.items():
            logger.info(f"{state}: {count} ({count/len(results)*100:.1f}%)")

        logger.info("\n" + "=" * 70)
        logger.info("最新状态：")
        logger.info("=" * 70)
        latest = results.iloc[-1]
        logger.info(f"日期: {latest['trade_date'].strftime('%Y-%m-%d')}")
        logger.info(f"状态: {latest['market_state']}")
        logger.info(f"动作: {latest['action']}")
        logger.info(f"建议仓位: {latest['position']:.1%}")

        logger.info("\n" + "=" * 70)
        logger.info("✓ 9指标风格模型分析完成！")
        logger.info("=" * 70)

        return results


def main():
    """主函数"""
    start_time = datetime.now().timestamp()
    failure_reason = None
    try:
        model = StyleModel9Indicators()
        results = model.run()

        if results is not None:
            print("\n" + "=" * 70)
            print("使用说明：")
            print("=" * 70)
            print("1. ACCUMULATION: 积累期，建议仓位50%")
            print("2. HEALTHY_UP: 健康扩张，建议仓位100%")
            print("3. DISTRIBUTION: 风险转移，建议仓位30%")
            print("4. EXIT: 逻辑失效，建议清仓")
    except Exception as exc:
        failure_reason = str(exc)
        raise
    finally:
        log_task_summary(
            logger,
            task_name="style_model_9indicators",
            window=None,
            elapsed_s=datetime.now().timestamp() - start_time,
            error=failure_reason,
        )


if __name__ == "__main__":
    main()
