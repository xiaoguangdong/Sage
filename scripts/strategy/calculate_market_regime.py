#!/usr/bin/env python3
"""
市场风格状态机

识别当前市场风格（趋势、流动性、基本面、投机），为量化因子选择提供依据
"""

import logging
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

from scripts.data._shared.runtime import get_data_path, get_tushare_root

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class MarketRegimeAnalyzer:
    """市场风格分析器"""

    def __init__(self):
        self.data_dir = str(get_tushare_root())
        self.output_dir = str(get_data_path("processed", "factors", ensure=True))

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info("市场风格状态机初始化")

    def load_data(self):
        """加载所有必要的数据"""
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

    def calculate_features(self):
        """计算所有风格特征"""
        logger.info("\n" + "=" * 70)
        logger.info("计算风格特征...")
        logger.info("=" * 70)

        # 1. 计算收益率
        logger.info("计算个股收益率...")
        daily_sorted = self.daily.sort_values(["ts_code", "trade_date"]).copy()
        daily_sorted["ret_1w"] = daily_sorted.groupby("ts_code")["close"].pct_change(5) * 100
        daily_sorted["ret_2w"] = daily_sorted.groupby("ts_code")["close"].pct_change(10) * 100
        daily_sorted["ret_4w"] = daily_sorted.groupby("ts_code")["close"].pct_change(20) * 100

        # 计算未来收益率（用于IC计算）
        daily_sorted["fwd_ret_1w"] = daily_sorted.groupby("ts_code")["close"].pct_change(-5) * 100
        daily_sorted["fwd_ret_2w"] = daily_sorted.groupby("ts_code")["close"].pct_change(-10) * 100
        daily_sorted["fwd_ret_4w"] = daily_sorted.groupby("ts_code")["close"].pct_change(-20) * 100
        daily_sorted["fwd_ret_12w"] = daily_sorted.groupby("ts_code")["close"].pct_change(-60) * 100

        # 计算波动率
        daily_sorted["vol_2w"] = daily_sorted.groupby("ts_code")["ret_1w"].transform(lambda x: x.rolling(10).std())

        # 计算换手率变化
        daily_sorted = daily_sorted.merge(
            self.basic[["ts_code", "trade_date", "turnover_rate"]], on=["ts_code", "trade_date"], how="left"
        )
        daily_sorted["turnover_chg_4w"] = daily_sorted.groupby("ts_code")["turnover_rate"].transform(
            lambda x: (x - x.shift(20)) / x.shift(20) * 100
        )

        # 合并财务数据
        logger.info("合并财务数据...")
        fina_subset = self.fina[["ts_code", "end_date", "roe"]].copy()
        fina_subset = fina_subset.rename(columns={"end_date": "trade_date"})
        daily_sorted = daily_sorted.merge(fina_subset, on=["ts_code", "trade_date"], how="left")
        daily_sorted["roe_ttm"] = daily_sorted.groupby("ts_code")["roe"].transform(lambda x: x.ffill())

        self.daily_features = daily_sorted

        # 计算每个日期的风格特征
        logger.info("计算每个日期的风格特征...")

        all_dates = sorted(self.daily_features["trade_date"].unique())
        # 只选择每周的数据
        weekly_dates = [d for i, d in enumerate(all_dates) if i % 5 == 0 and i >= 20]

        regime_features = []

        for i, trade_date in enumerate(weekly_dates):
            if (i + 1) % 10 == 0:
                logger.info(f"进度: {i+1}/{len(weekly_dates)}")

            # 获取该日期的数据
            date_data = self.daily_features[
                (self.daily_features["trade_date"] == trade_date)
                & (self.daily_features["ret_4w"].notna())
                & (self.daily_features["fwd_ret_4w"].notna())
            ].copy()

            if len(date_data) < 100:  # 需要足够的数据
                continue

            # Feature 1: 横截面动量有效性 (IC)
            valid_data = date_data[["ret_4w", "fwd_ret_4w"]].dropna()
            if len(valid_data) > 30:
                mom_ic, _ = stats.spearmanr(valid_data["ret_4w"], valid_data["fwd_ret_4w"])
            else:
                mom_ic = np.nan

            # Feature 2: 成交额变化解释力
            valid_data = date_data[["turnover_chg_4w", "fwd_ret_2w"]].dropna()
            if len(valid_data) > 30:
                liq_ic, _ = stats.spearmanr(valid_data["turnover_chg_4w"], valid_data["fwd_ret_2w"])
            else:
                liq_ic = np.nan

            # Feature 3: 波动率-收益相关性
            valid_data = date_data[["vol_2w", "fwd_ret_2w"]].dropna()
            if len(valid_data) > 30:
                vol_ret_corr, _ = stats.spearmanr(valid_data["vol_2w"], valid_data["fwd_ret_2w"])
            else:
                vol_ret_corr = np.nan

            # Feature 4: 基本面解释力 (ROE IC)
            valid_data = date_data[["roe_ttm", "fwd_ret_12w"]].dropna()
            if len(valid_data) > 30:
                roe_ic, _ = stats.spearmanr(valid_data["roe_ttm"], valid_data["fwd_ret_12w"])
            else:
                roe_ic = np.nan

            # Feature 5: 收益集中度
            date_data["ret_4w_abs"] = date_data["ret_4w"].abs()
            total_ret_sum = date_data["ret_4w_abs"].sum()
            top10_ret_sum = date_data.nlargest(10, "ret_4w_abs")["ret_4w_abs"].sum()
            ret_concentration = top10_ret_sum / total_ret_sum if total_ret_sum > 0 else np.nan

            # Feature 6: 大小盘相对强度
            # 使用市值区分大小盘
            date_data = date_data.merge(
                self.basic[["ts_code", "trade_date", "total_mv"]], on=["ts_code", "trade_date"], how="left"
            )
            median_mv = date_data["total_mv"].median()
            small_cap_ret = date_data[date_data["total_mv"] < median_mv]["fwd_ret_2w"].mean()
            large_cap_ret = date_data[date_data["total_mv"] >= median_mv]["fwd_ret_2w"].mean()
            small_big = small_cap_ret - large_cap_ret

            # Feature 7: 高换手股票胜率
            turnover_rank = date_data["turnover_rate"].rank(pct=True)
            high_turn_ret = date_data[turnover_rank > 0.8]["fwd_ret_1w"].mean()

            # Feature 8: 行业轮动速度（简化版：使用收益率标准差）
            ret_std = date_data["ret_4w"].std()
            industry_switch = ret_std / 10  # 标准化

            # Feature 9: 市场趋势强度（使用指数）
            hs300_data = self.index_data[self.index_data["code"] == "000300.SH"].copy()
            hs300_data = hs300_data.sort_values("trade_date")
            hs300_data["ma_20w"] = hs300_data["close"].rolling(100).mean()
            hs300_data = hs300_data.dropna()

            idx_data = hs300_data[hs300_data["trade_date"] <= trade_date].tail(1)
            if len(idx_data) > 0:
                index_trend = (
                    (idx_data["close"].values[0] - idx_data["ma_20w"].values[0]) / idx_data["ma_20w"].values[0] * 100
                )
            else:
                index_trend = np.nan

            # Feature 10: 横截面相关性
            valid_rets = date_data["fwd_ret_1w"].dropna()
            if len(valid_rets) > 50:
                # 计算所有股票之间的相关性（采样）
                sample_rets = valid_rets.sample(min(100, len(valid_rets)))
                corr_matrix = np.corrcoef(sample_rets)
                # 取上三角矩阵的平均
                mean_pairwise_corr = np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
            else:
                mean_pairwise_corr = np.nan

            # 计算风格维度
            trend_raw = mom_ic * 1.0 + index_trend * 0.6 - industry_switch * 0.3

            liquidity_raw = liq_ic * 1.0 + small_big * 0.5 + high_turn_ret * 0.5

            fundamental_raw = roe_ic * 1.2 - ret_concentration * 0.5

            speculative_raw = vol_ret_corr * 0.8 + ret_concentration * 1.0 + industry_switch * 0.5

            regime_features.append(
                {
                    "trade_date": trade_date,
                    "feature_1_mom_ic": mom_ic,
                    "feature_2_liq_ic": liq_ic,
                    "feature_3_vol_ret_corr": vol_ret_corr,
                    "feature_4_roe_ic": roe_ic,
                    "feature_5_ret_concentration": ret_concentration,
                    "feature_6_small_big": small_big,
                    "feature_7_high_turn_ret": high_turn_ret,
                    "feature_8_industry_switch": industry_switch,
                    "feature_9_index_trend": index_trend,
                    "feature_10_mean_pairwise_corr": mean_pairwise_corr,
                    "trend_raw": trend_raw,
                    "liquidity_raw": liquidity_raw,
                    "fundamental_raw": fundamental_raw,
                    "speculative_raw": speculative_raw,
                }
            )

        self.regime_df = pd.DataFrame(regime_features)
        logger.info(f"✓ 风格特征计算完成: {len(self.regime_df)} 条记录")

        return self.regime_df

    def save_results(self):
        """保存结果"""
        logger.info("\n" + "=" * 70)
        logger.info("保存结果...")
        logger.info("=" * 70)

        # 保存风格特征数据
        output_file = os.path.join(self.output_dir, "market_regime_features.parquet")
        self.regime_df.to_parquet(output_file)
        logger.info(f"✓ 风格特征: {output_file}")

        # 保存CSV格式
        csv_file = os.path.join(self.output_dir, "market_regime_features.csv")
        self.regime_df.to_csv(csv_file, index=False)
        logger.info(f"✓ 风格特征CSV: {csv_file}")

        # 输出统计摘要
        logger.info("\n" + "=" * 70)
        logger.info("风格特征统计摘要：")
        logger.info("=" * 70)

        style_cols = ["trend_raw", "liquidity_raw", "fundamental_raw", "speculative_raw"]
        for col in style_cols:
            valid_data = self.regime_df[col].dropna()
            logger.info(f"\n{col}:")
            logger.info(f"  平均值: {valid_data.mean():.4f}")
            logger.info(f"  标准差: {valid_data.std():.4f}")
            logger.info(f"  最小值: {valid_data.min():.4f}")
            logger.info(f"  最大值: {valid_data.max():.4f}")
            logger.info(f"  中位数: {valid_data.median():.4f}")

        # 显示最新的风格状态
        logger.info("\n" + "=" * 70)
        logger.info("最新风格状态（最新10个周期）：")
        logger.info("=" * 70)
        latest_data = self.regime_df.tail(10)[
            ["trade_date", "trend_raw", "liquidity_raw", "fundamental_raw", "speculative_raw"]
        ]
        print(latest_data.to_string(index=False))

    def run(self):
        """执行完整的风格分析流程"""
        logger.info("\n" + "=" * 70)
        logger.info("开始分析市场风格...")
        logger.info("=" * 70)

        if not self.load_data():
            logger.error("数据加载失败")
            return None

        if not self.calculate_features():
            logger.error("特征计算失败")
            return None

        self.save_results()

        logger.info("\n" + "=" * 70)
        logger.info("✓ 市场风格分析完成！")
        logger.info("=" * 70)

        return self.regime_df


def main():
    """主函数"""
    analyzer = MarketRegimeAnalyzer()
    regime_df = analyzer.run()

    if regime_df is not None:
        print("\n" + "=" * 70)
        print("使用说明：")
        print("=" * 70)
        print("1. trend_raw > 0: 趋势市，可以使用动量因子")
        print("2. liquidity_raw > 0: 流动性驱动，关注换手率")
        print("3. fundamental_raw > 0: 基本面有效，关注ROE等基本面指标")
        print("4. speculative_raw > 0: 投机情绪高涨，关注高波动、小盘股")
        print("\n根据这些风格维度，动态调整因子权重：")
        print("- 趋势市: 增加动量因子权重")
        print("- 基本面市: 增加质量因子权重")
        print("- 投机市: 关注小盘、高波动股票")


if __name__ == "__main__":
    main()
