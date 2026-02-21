#!/usr/bin/env python3
"""
观察推导风格模型（GLM方案）

基于观察资金流向，逆向推导市场逻辑，动态调整因子权重
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd

from scripts.data._shared.runtime import get_data_path, get_tushare_root, log_task_summary, setup_logger

logger = setup_logger("style_model_observation", module="models")


class StyleModelObservation:
    """观察推导风格模型"""

    def __init__(self):
        self.data_dir = str(get_tushare_root())
        self.output_dir = str(get_data_path("processed", "factors", ensure=True))

        # 逻辑类型定义
        self.LOGIC_TYPES = {
            "momentum": "动量驱动",
            "fundamental": "基本面驱动",
            "small_cap": "小盘投机",
            "institution": "机构抱团",
        }

        # 警告级别
        self.WARNING_LEVELS = {
            "GREEN": "逻辑健康，继续执行",
            "YELLOW": "逻辑弱化，建议降仓",
            "RED": "逻辑失效，建议清仓",
        }

        logger.info("观察推导风格模型初始化")

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

        # 加载财务数据
        logger.info("加载财务数据...")
        fina_data = []
        for year in range(2020, 2026):
            file_path = os.path.join(self.data_dir, "fundamental", f"fina_indicator_{year}.parquet")
            if os.path.exists(file_path):
                df = pd.read_parquet(file_path)
                fina_data.append(df)

        if fina_data:
            self.fina = pd.concat(fina_data, ignore_index=True)
            self.fina["end_date"] = pd.to_datetime(self.fina["end_date"])
            logger.info(f"✓ 财务数据总计: {len(self.fina)} 条记录")
        else:
            logger.error("✗ 未找到财务数据")
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

    def find_hot_sectors(self, date_data, window=20):
        """找到热门板块"""
        # 简化版：使用成交额最大的股票作为代理
        date_data_sorted = date_data.sort_values("amount", ascending=False)
        top_stocks = date_data_sorted.head(100)

        # 计算这些股票的特征
        hot_features = {
            "avg_ret_20d": top_stocks["pct_chg"].mean(),
            "avg_turnover": top_stocks["turnover_rate"].mean(),
            "avg_vol": top_stocks["vol"].mean(),
            "stock_count": len(top_stocks),
        }

        return hot_features

    def analyze_winners(self, hot_features, date_data):
        """分析热门股票的特征"""
        # 获取赢家股票（涨幅前20%）
        date_data_sorted = date_data.sort_values("pct_chg", ascending=False)
        top_20pct_count = int(len(date_data_sorted) * 0.2)
        winners = date_data_sorted.head(top_20pct_count)

        # 获取输家股票（涨幅后20%）
        losers = date_data_sorted.tail(top_20pct_count)

        # 特征维度1：基本面特征
        fundamental_features = {
            "winners_roe": winners.get("roe", np.nan).mean(),
            "winners_pe": winners.get("pe", np.nan).mean(),
            "winners_market_cap": winners.get("total_mv", np.nan).median(),
            "losers_roe": losers.get("roe", np.nan).mean(),
            "losers_pe": losers.get("pe", np.nan).mean(),
            "losers_market_cap": losers.get("total_mv", np.nan).median(),
        }

        # 特征维度2：技术特征
        technical_features = {
            "winners_momentum": winners["pct_chg"].mean(),
            "winners_volatility": winners["pct_chg"].std(),
            "winners_turnover": winners["turnover_rate"].mean(),
            "losers_momentum": losers["pct_chg"].mean(),
            "losers_volatility": losers["pct_chg"].std(),
            "losers_turnover": losers["turnover_rate"].mean(),
        }

        return {"fundamental": fundamental_features, "technical": technical_features, "hot_features": hot_features}

    def infer_logic_from_features(self, winners_features):
        """推断当前市场逻辑"""
        logic_scores = {}

        f = winners_features["fundamental"]
        t = winners_features["technical"]

        # 逻辑1：基本面驱动
        roe_diff = f["winners_roe"] - f["losers_roe"]
        if roe_diff > 0.05:  # 赢家ROE比输家高5%以上
            logic_scores["fundamental"] = 0.8
        else:
            logic_scores["fundamental"] = 0.3

        # 逻辑2：动量驱动
        if t["winners_momentum"] > 0:
            logic_scores["momentum"] = 0.9
        else:
            logic_scores["momentum"] = 0.4

        # 逻辑3：小盘投机
        cap_ratio = f["winners_market_cap"] / (f["losers_market_cap"] + 1e-8)
        if cap_ratio < 0.8:  # 赢家市值比输家小
            logic_scores["small_cap"] = 0.7
        else:
            logic_scores["small_cap"] = 0.3

        # 逻辑4：机构抱团（简化：低波动+低换手）
        if t["winners_volatility"] < t["losers_volatility"] * 0.8:
            logic_scores["institution"] = 0.6
        else:
            logic_scores["institution"] = 0.3

        # 选择主导逻辑
        dominant_logic = max(logic_scores, key=logic_scores.get)

        return {"logic_type": dominant_logic, "logic_score": logic_scores, "confidence": logic_scores[dominant_logic]}

    def dynamic_factor_weights(self, market_logic):
        """根据当前市场逻辑，动态调整因子权重"""
        base_weights = {"momentum": 0.25, "quality": 0.25, "liquidity": 0.25, "risk": 0.25}

        logic_type = market_logic["logic_type"]

        # 逻辑1：动量驱动 → 提高动量因子权重
        if logic_type == "momentum":
            base_weights["momentum"] = 0.5
            base_weights["quality"] = 0.2
            base_weights["liquidity"] = 0.2
            base_weights["risk"] = 0.1

        # 逻辑2：基本面驱动 → 提高质量因子权重
        elif logic_type == "fundamental":
            base_weights["quality"] = 0.5
            base_weights["momentum"] = 0.2
            base_weights["liquidity"] = 0.2
            base_weights["risk"] = 0.1

        # 逻辑3：小盘投机 → 降低质量因子权重，提高流动性
        elif logic_type == "small_cap":
            base_weights["momentum"] = 0.4
            base_weights["liquidity"] = 0.4
            base_weights["quality"] = 0.1
            base_weights["risk"] = 0.1

        # 逻辑4：机构抱团 → 平衡权重
        elif logic_type == "institution":
            base_weights["momentum"] = 0.3
            base_weights["quality"] = 0.3
            base_weights["liquidity"] = 0.2
            base_weights["risk"] = 0.2

        return base_weights

    def monitor_logic_validity(self, historical_logics, current_logic, window=20):
        """监控逻辑有效性"""
        if len(historical_logics) < 5:
            return {"validity_score": 0.7, "logic_return_decay": 0.0, "logic_drift": 0.0, "logic_consistency": 0.7}

        # 计算逻辑收益衰减（简化版：使用逻辑置信度的变化）
        recent_confidences = [lg["confidence"] for lg in historical_logics[-10:]]
        if len(recent_confidences) > 0:
            logic_return_decay = max(0, recent_confidences[0] - current_logic["confidence"])
        else:
            logic_return_decay = 0.0

        # 计算逻辑漂移（简化版：逻辑类型的变化）
        logic_types = [lg["logic_type"] for lg in historical_logics[-5:]]
        current_type = current_logic["logic_type"]
        logic_drift = 1.0 - (logic_types.count(current_type) / len(logic_types))

        # 计算逻辑一致性
        logic_consistency = 1.0 - logic_drift

        # 综合评分
        validity_score = (
            (1 - logic_return_decay) * 0.3 + (1 - logic_drift) * 0.3 + logic_consistency * 0.2 + 0.2  # 基础分
        )

        return {
            "validity_score": validity_score,
            "logic_return_decay": logic_return_decay,
            "logic_drift": logic_drift,
            "logic_consistency": logic_consistency,
        }

    def logic_failure_warning(self, validity_metrics):
        """判定逻辑失效"""
        if validity_metrics["validity_score"] > 0.7:
            return {
                "warning_level": "GREEN",
                "action": "CONTINUE",
                "message": self.WARNING_LEVELS["GREEN"],
                "position_adjustment": 0.0,
            }

        elif validity_metrics["validity_score"] > 0.5:
            return {
                "warning_level": "YELLOW",
                "action": "REDUCE",
                "message": self.WARNING_LEVELS["YELLOW"],
                "position_adjustment": -0.3,
            }

        else:
            return {
                "warning_level": "RED",
                "action": "EXIT",
                "message": self.WARNING_LEVELS["RED"],
                "position_adjustment": -1.0,
            }

    def decision_engine(self, current_logic, validity_metrics, market_state):
        """决策引擎"""
        warning = self.logic_failure_warning(validity_metrics)

        if warning["warning_level"] == "RED":
            # 逻辑失效，清仓
            return {"position": 0.0, "action": "CLEAR_POSITION", "reason": warning["message"]}

        elif warning["warning_level"] == "YELLOW":
            # 逻辑弱化，降仓
            base_position = 0.6  # 震荡市基础仓位
            if market_state == "BULL":
                base_position = 0.9
            elif market_state == "BEAR":
                base_position = 0.3

            return {
                "position": base_position * 0.7,
                "action": "REDUCE_POSITION",
                "reason": warning["message"],
                "current_logic": current_logic["logic_type"],
                "validity_score": validity_metrics["validity_score"],
            }

        else:
            # 逻辑健康，正常参与
            base_position = 0.6
            if market_state == "BULL":
                base_position = 0.9
            elif market_state == "BEAR":
                base_position = 0.3

            return {
                "position": base_position,
                "action": "PARTICIPATE",
                "current_logic": current_logic["logic_type"],
                "validity_score": validity_metrics["validity_score"],
                "confidence": current_logic["confidence"],
            }

    def run(self):
        """执行完整的分析流程"""
        logger.info("\n" + "=" * 70)
        logger.info("开始观察推导风格模型分析...")
        logger.info("=" * 70)

        if not self.load_data():
            logger.error("数据加载失败")
            return None

        # 计算市场层面的指标（使用沪深300作为代理）
        logger.info("计算市场层面指标...")
        hs300 = self.index_data[self.index_data["code"] == "000300.SH"].copy()
        hs300 = hs300.sort_values("trade_date")

        # 合并daily_basic数据
        hs300_basic = self.basic[self.basic["ts_code"] == "000300.SH"].copy()
        hs300_basic = hs300_basic.sort_values("trade_date")
        hs300 = hs300.merge(hs300_basic[["trade_date", "turnover_rate", "pe", "total_mv"]], on="trade_date", how="left")

        # 使用全市场数据
        market_daily = self.daily.copy()
        market_daily = market_daily.merge(
            self.basic[["ts_code", "trade_date", "turnover_rate", "pe", "total_mv"]],
            on=["ts_code", "trade_date"],
            how="left",
        )

        results = []
        historical_logics = []

        # 按日期处理
        all_dates = sorted(market_daily["trade_date"].unique())
        weekly_dates = [d for i, d in enumerate(all_dates) if i % 5 == 0 and i >= 20]

        logger.info(f"处理 {len(weekly_dates)} 个交易日期...")

        for i, trade_date in enumerate(weekly_dates):
            if (i + 1) % 20 == 0:
                logger.info(f"进度: {i+1}/{len(weekly_dates)}")

            # 获取当日数据
            date_data = market_daily[market_daily["trade_date"] == trade_date].copy()

            if len(date_data) < 100:
                continue

            # 步骤1：发现当前市场逻辑
            hot_features = self.find_hot_sectors(date_data)
            winners_features = self.analyze_winners(hot_features, date_data)
            current_logic = self.infer_logic_from_features(winners_features)

            # 步骤2：监控逻辑有效性
            validity_metrics = self.monitor_logic_validity(historical_logics, current_logic)

            # 步骤3：决策（简化：假设市场状态为震荡）
            market_state = "SIDEWAYS"
            decision = self.decision_engine(current_logic, validity_metrics, market_state)

            # 记录结果
            results.append(
                {
                    "trade_date": trade_date,
                    "logic_type": current_logic["logic_type"],
                    "logic_momentum": current_logic["logic_score"]["momentum"],
                    "logic_fundamental": current_logic["logic_score"]["fundamental"],
                    "logic_small_cap": current_logic["logic_score"]["small_cap"],
                    "logic_institution": current_logic["logic_score"]["institution"],
                    "confidence": current_logic["confidence"],
                    "validity_score": validity_metrics["validity_score"],
                    "logic_return_decay": validity_metrics["logic_return_decay"],
                    "logic_drift": validity_metrics["logic_drift"],
                    "logic_consistency": validity_metrics["logic_consistency"],
                    "action": decision["action"],
                    "position": decision["position"],
                }
            )

            # 保存历史逻辑
            historical_logics.append(current_logic)

        # 转换为DataFrame
        results_df = pd.DataFrame(results)

        # 保存结果
        output_file = os.path.join(self.output_dir, "style_model_observation_results.parquet")
        results_df.to_parquet(output_file)
        logger.info(f"✓ 结果已保存: {output_file}")

        # 输出统计摘要
        logger.info("\n" + "=" * 70)
        logger.info("逻辑分布统计：")
        logger.info("=" * 70)
        logic_counts = results_df["logic_type"].value_counts()
        for logic, count in logic_counts.items():
            logger.info(f"{logic}: {count} ({count/len(results_df)*100:.1f}%)")

        logger.info("\n" + "=" * 70)
        logger.info("最新状态：")
        logger.info("=" * 70)
        latest = results_df.iloc[-1]
        logger.info(f"日期: {latest['trade_date'].strftime('%Y-%m-%d')}")
        logger.info(f"当前逻辑: {latest['logic_type']}")
        logger.info(f"逻辑置信度: {latest['confidence']:.2f}")
        logger.info(f"有效性评分: {latest['validity_score']:.2f}")
        logger.info(f"动作: {latest['action']}")
        logger.info(f"建议仓位: {latest['position']:.1%}")

        logger.info("\n" + "=" * 70)
        logger.info("✓ 观察推导风格模型分析完成！")
        logger.info("=" * 70)

        return results_df


def main():
    """主函数"""
    start_time = datetime.now().timestamp()
    failure_reason = None
    try:
        model = StyleModelObservation()
        results = model.run()

        if results is not None:
            print("\n" + "=" * 70)
            print("使用说明：")
            print("=" * 70)
            print("1. momentum: 动量驱动，提高动量因子权重")
            print("2. fundamental: 基本面驱动，提高质量因子权重")
            print("3. small_cap: 小盘投机，降低质量因子权重")
            print("4. institution: 机构抱团，平衡权重")
            print("\n警告级别：")
            print("- GREEN: 逻辑健康，继续执行")
            print("- YELLOW: 逻辑弱化，建议降仓")
            print("- RED: 逻辑失效，建议清仓")
    except Exception as exc:
        failure_reason = str(exc)
        raise
    finally:
        log_task_summary(
            logger,
            task_name="style_model_observation",
            window=None,
            elapsed_s=datetime.now().timestamp() - start_time,
            error=failure_reason,
        )


if __name__ == "__main__":
    main()
