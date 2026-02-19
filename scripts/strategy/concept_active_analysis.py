#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
概念活跃度分析系统
计算2024-09-24到2026-02-06期间的活跃概念，识别牛市主线逻辑

核心功能：
1. 计算期间活跃概念（使用全部概念）
2. 每周更新概念列表和成分股
3. 识别牛市主线逻辑

使用说明：
python concept_active_analysis.py --analyze  # 分析活跃概念
python concept_active_analysis.py --update  # 更新数据
python concept_active_analysis.py --mainline  # 识别主线逻辑
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data._shared.runtime import get_tushare_root, setup_logger

logger = setup_logger(Path(__file__).stem, module="strategy")

# 配置参数
TIMEOUT = 30
DATA_DIR = str(get_tushare_root() / "sectors")
START_DATE = "20240924"
END_DATE = "20260206"


class ConceptActiveAnalyzer:
    """概念活跃度分析器"""

    def __init__(self, token=None, timeout=TIMEOUT):
        self.timeout = timeout
        self.data_dir = DATA_DIR
        self.market = "EB"

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs("logs", exist_ok=True)

    def fetch_concept_history(self, start_date, end_date):
        """获取概念历史数据"""
        logger.info(f"读取概念历史数据: {start_date} ~ {end_date}")

        data_path = Path(self.data_dir) / "ths_daily.parquet"
        if not data_path.exists():
            logger.error("未找到 ths_daily 数据，请先运行 tushare_downloader.py --task ths_daily")
            return None

        df = pd.read_parquet(data_path)
        if df.empty:
            logger.error("ths_daily 数据为空")
            return None

        df = df.copy()
        df["trade_date"] = df["trade_date"].astype(str)
        start_str = pd.Timestamp(start_date).strftime("%Y%m%d")
        end_str = pd.Timestamp(end_date).strftime("%Y%m%d")
        df = df[(df["trade_date"] >= start_str) & (df["trade_date"] <= end_str)]
        logger.info(f"过滤后记录数: {len(df)}")
        return df

    def calculate_concept_activity(self, data):
        """计算概念活跃度"""
        logger.info("计算概念活跃度...")

        # 转换日期
        data["trade_date"] = pd.to_datetime(data["trade_date"])

        # 计算每个概念的综合活跃度指标
        concept_activity = []

        for concept_code in data["ts_code"].unique():
            concept_data = data[data["ts_code"] == concept_code].copy()
            concept_data = concept_data.sort_values("trade_date")

            # 基础信息
            concept_name = concept_data["name"].iloc[0]
            total_days = len(concept_data)

            # 1. 累计涨幅
            cumulative_return = (1 + concept_data["pct_change"] / 100).prod() - 1

            # 2. 日均涨幅
            avg_return = concept_data["pct_change"].mean()

            # 3. 最大回撤
            cumulative_series = (1 + concept_data["pct_change"] / 100).cumprod()
            max_drawdown = (cumulative_series / cumulative_series.cummax() - 1).min() * 100

            # 4. 波动率
            volatility = concept_data["pct_change"].std()

            # 5. 上涨天数
            up_days = (concept_data["pct_change"] > 0).sum()
            up_ratio = up_days / total_days

            # 6. 涨幅>5%的天数（强势日）
            strong_days = (concept_data["pct_change"] > 5).sum()
            strong_ratio = strong_days / total_days

            # 7. 涨幅<-5%的天数（弱势日）
            weak_days = (concept_data["pct_change"] < -5).sum()
            weak_ratio = weak_days / total_days

            # 8. 领涨次数（涨跌幅排名前10%）
            # 计算每日排名
            concept_data["daily_rank"] = concept_data.groupby("trade_date")["pct_change"].rank(
                ascending=False, pct=True
            )
            leading_days = (concept_data["daily_rank"] <= 0.1).sum()
            leading_ratio = leading_days / total_days

            # 9. 平均换手率
            avg_turnover = concept_data["turnover_rate"].mean()

            # 10. 连续上涨最大天数
            concept_data["up_streak"] = (concept_data["pct_change"] > 0).astype(int)
            concept_data["streak_id"] = (concept_data["up_streak"].diff() != 0).cumsum()
            max_up_streak = concept_data[concept_data["up_streak"] == 1].groupby("streak_id").size().max()

            concept_activity.append(
                {
                    "ts_code": concept_code,
                    "name": concept_name,
                    "total_days": total_days,
                    "cumulative_return": cumulative_return * 100,
                    "avg_return": avg_return,
                    "max_drawdown": max_drawdown,
                    "volatility": volatility,
                    "up_ratio": up_ratio,
                    "strong_ratio": strong_ratio,
                    "weak_ratio": weak_ratio,
                    "leading_ratio": leading_ratio,
                    "avg_turnover": avg_turnover,
                    "max_up_streak": max_up_streak,
                }
            )

        activity_df = pd.DataFrame(concept_activity)

        # 计算综合活跃度评分
        activity_df = self.calculate_activity_score(activity_df)

        logger.info(f"活跃度计算完成: {len(activity_df)} 个概念")

        return activity_df

    def calculate_activity_score(self, activity_df):
        """计算综合活跃度评分"""
        logger.info("计算综合活跃度评分...")

        # 标准化指标
        def normalize_score(series, reverse=False):
            min_val = series.min()
            max_val = series.max()
            if max_val == min_val:
                return pd.Series([50] * len(series), index=series.index)
            if reverse:
                return (series - min_val) / (max_val - min_val) * 100
            else:
                return (max_val - series) / (max_val - min_val) * 100

        # 各维度评分
        activity_df["return_score"] = normalize_score(activity_df["cumulative_return"], reverse=True)
        activity_df["up_ratio_score"] = activity_df["up_ratio"] * 100
        activity_df["strong_ratio_score"] = activity_df["strong_ratio"] * 100
        activity_df["leading_ratio_score"] = activity_df["leading_ratio"] * 100
        activity_df["turnover_score"] = normalize_score(activity_df["avg_turnover"], reverse=True)
        activity_df["streak_score"] = normalize_score(activity_df["max_up_streak"], reverse=True)

        # 综合评分（权重）
        weights = {
            "return_score": 0.30,  # 累计涨幅
            "up_ratio_score": 0.20,  # 上涨天数比例
            "strong_ratio_score": 0.15,  # 强势日比例
            "leading_ratio_score": 0.15,  # 领涨次数比例
            "turnover_score": 0.10,  # 换手率
            "streak_score": 0.10,  # 连续上涨天数
        }

        activity_df["activity_score"] = (
            activity_df["return_score"] * weights["return_score"]
            + activity_df["up_ratio_score"] * weights["up_ratio_score"]
            + activity_df["strong_ratio_score"] * weights["strong_ratio_score"]
            + activity_df["leading_ratio_score"] * weights["leading_ratio_score"]
            + activity_df["turnover_score"] * weights["turnover_score"]
            + activity_df["streak_score"] * weights["streak_score"]
        )

        # 排序
        activity_df = activity_df.sort_values("activity_score", ascending=False)

        # 活跃度等级
        activity_df["activity_level"] = pd.cut(
            activity_df["activity_score"], bins=[0, 30, 50, 70, 100], labels=["低活跃", "中活跃", "高活跃", "超高活跃"]
        )

        return activity_df

    def identify_bull_market_mainline(self, activity_df, data):
        """识别牛市主线逻辑"""
        logger.info("识别牛市主线逻辑...")

        # 按时间划分阶段
        data["trade_date"] = pd.to_datetime(data["trade_date"])
        data["month"] = data["trade_date"].dt.to_period("M")

        # 1. 找出每个阶段的活跃概念
        monthly_top_concepts = []
        for month in sorted(data["month"].unique()):
            month_data = data[data["month"] == month]
            month_top = month_data.groupby("ts_code")["pct_change"].mean().sort_values(ascending=False).head(20)

            for concept_code, pct in month_top.items():
                concept_name = data[data["ts_code"] == concept_code]["name"].iloc[0]
                monthly_top_concepts.append(
                    {"month": str(month), "ts_code": concept_code, "name": concept_name, "monthly_return": pct}
                )

        monthly_df = pd.DataFrame(monthly_top_concepts)

        # 2. 统计概念上榜次数
        concept_appearance = monthly_df["ts_code"].value_counts().head(30)

        # 3. 识别主线候选（上榜次数>=3且累计涨幅>20%）
        mainline_candidates = []
        for concept_code in concept_appearance.index:
            if concept_appearance[concept_code] >= 3:
                concept_info = activity_df[activity_df["ts_code"] == concept_code]
                if len(concept_info) > 0 and concept_info["cumulative_return"].iloc[0] > 20:
                    mainline_candidates.append(
                        {
                            "ts_code": concept_code,
                            "name": concept_info["name"].iloc[0],
                            "appearance_count": concept_appearance[concept_code],
                            "cumulative_return": concept_info["cumulative_return"].iloc[0],
                            "activity_score": concept_info["activity_score"].iloc[0],
                        }
                    )

        mainline_df = pd.DataFrame(mainline_candidates)
        mainline_df = mainline_df.sort_values("activity_score", ascending=False)

        # 4. 主线分类
        mainline_df["mainline_category"] = self.classify_mainline(mainline_df)

        logger.info(f"主线识别完成: {len(mainline_df)} 个主线候选")

        return mainline_df, monthly_df

    def classify_mainline(self, mainline_df):
        """分类主线"""
        categories = []

        for idx, row in mainline_df.iterrows():
            name = row["name"]

            # 根据关键词分类
            if any(
                keyword in name
                for keyword in ["芯片", "半导体", "集成电路", "电子", "通信", "5G", "6G", "光刻", "存储"]
            ):
                category = "科技主线"
            elif any(keyword in name for keyword in ["新能源", "锂电", "光伏", "风电", "储能", "电池", "充电"]):
                category = "新能源主线"
            elif any(keyword in name for keyword in ["医药", "生物", "疫苗", "医疗", "健康", "抗疫"]):
                category = "医药主线"
            elif any(keyword in name for keyword in ["军工", "国防", "航天", "航空", "导弹"]):
                category = "军工主线"
            elif any(keyword in name for keyword in ["消费", "零售", "食品", "白酒", "家电"]):
                category = "消费主线"
            elif any(keyword in name for keyword in ["金融", "银行", "证券", "保险", "信托"]):
                category = "金融主线"
            elif any(keyword in name for keyword in ["地产", "建筑", "基建", "房地产"]):
                category = "地产主线"
            elif any(keyword in name for keyword in ["汽车", "整车", "零部件", "智能驾驶"]):
                category = "汽车主线"
            else:
                category = "其他主线"

            categories.append(category)

        return categories

    def save_results(self, activity_df, mainline_df=None, monthly_df=None, suffix=""):
        """保存结果"""
        timestamp = datetime.now().strftime("%Y%m%d")

        # 保存活跃度数据
        activity_path = os.path.join(self.data_dir, f"concept_activity_{timestamp}{suffix}.csv")
        activity_df.to_csv(activity_path, index=False, encoding="utf-8-sig")
        logger.info(f"活跃度数据已保存: {activity_path}")

        # 保存主线数据
        if mainline_df is not None:
            mainline_path = os.path.join(self.data_dir, f"concept_mainline_{timestamp}{suffix}.csv")
            mainline_df.to_csv(mainline_path, index=False, encoding="utf-8-sig")
            logger.info(f"主线数据已保存: {mainline_path}")

        # 保存月度数据
        if monthly_df is not None:
            monthly_path = os.path.join(self.data_dir, f"concept_monthly_{timestamp}{suffix}.csv")
            monthly_df.to_csv(monthly_path, index=False, encoding="utf-8-sig")
            logger.info(f"月度数据已保存: {monthly_path}")

        return activity_path, mainline_path, monthly_path

    def print_summary(self, activity_df, mainline_df):
        """打印分析摘要"""
        print("\n" + "=" * 100)
        print("概念活跃度分析摘要")
        print("=" * 100)

        print("\n1. 活跃概念Top 20:")
        print(f"{'排名':<4} {'概念名称':<20} {'活跃度评分':<12} {'累计涨幅':<12} {'上涨天数':<12} {'领涨次数':<12}")
        print("-" * 100)
        for i, row in activity_df.head(20).iterrows():
            print(
                f"{i+1:<4} {row['name']:<20} {row['activity_score']:>11.1f} {row['cumulative_return']:>11.2f}% {row['up_ratio']*100:>11.1f}% {row['leading_ratio']*100:>11.1f}%"
            )

        print("\n2. 主线逻辑Top 10:")
        print(f"{'排名':<4} {'概念名称':<20} {'主线分类':<12} {'上榜次数':<12} {'累计涨幅':<12} {'活跃度评分':<12}")
        print("-" * 100)
        for i, row in mainline_df.head(10).iterrows():
            print(
                f"{i+1:<4} {row['name']:<20} {row['mainline_category']:<12} {row['appearance_count']:>11} {row['cumulative_return']:>11.2f}% {row['activity_score']:>11.1f}"
            )

        print("\n3. 主线分类统计:")
        category_stats = mainline_df["mainline_category"].value_counts()
        for category, count in category_stats.items():
            print(f"   {category}: {count} 个")

        print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(description="概念活跃度分析系统")
    parser.add_argument("--analyze", action="store_true", help="分析活跃概念")
    parser.add_argument("--update", action="store_true", help="更新数据")
    parser.add_argument("--mainline", action="store_true", help="识别主线逻辑")
    parser.add_argument("--start-date", type=str, default=START_DATE, help="开始日期")
    parser.add_argument("--end-date", type=str, default=END_DATE, help="结束日期")

    args = parser.parse_args()

    analyzer = ConceptActiveAnalyzer()

    if args.analyze:
        logger.info("=" * 80)
        logger.info("分析概念活跃度")
        logger.info("=" * 80)

        # 获取历史数据
        data = analyzer.fetch_concept_history(args.start_date, args.end_date)

        if data is not None:
            # 保存原始数据
            analyzer.save_results(data, suffix="_raw")

            # 计算活跃度
            activity_df = analyzer.calculate_concept_activity(data)

            # 识别主线
            mainline_df, monthly_df = analyzer.identify_bull_market_mainline(activity_df, data)

            # 保存结果
            analyzer.save_results(activity_df, mainline_df, monthly_df)

            # 打印摘要
            analyzer.print_summary(activity_df, mainline_df)

    elif args.update:
        logger.info("=" * 80)
        logger.info("更新数据")
        logger.info("=" * 80)

        # 获取最近一个月的数据
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")

        data = analyzer.fetch_concept_history(start_date, end_date)

        if data is not None:
            analyzer.save_results(data, suffix="_update")
            logger.info("数据更新完成")

    elif args.mainline:
        logger.info("=" * 80)
        logger.info("识别主线逻辑")
        logger.info("=" * 80)

        # 加载活跃度数据
        activity_path = os.path.join(analyzer.data_dir, "concept_activity.csv")
        if os.path.exists(activity_path):
            activity_df = pd.read_csv(activity_path)

            # 加载原始数据
            data_path = os.path.join(analyzer.data_dir, "concept_daily_eastmoney.csv")
            if os.path.exists(data_path):
                data = pd.read_csv(data_path)

                # 识别主线
                mainline_df, monthly_df = analyzer.identify_bull_market_mainline(activity_df, data)

                # 保存结果
                analyzer.save_results(activity_df, mainline_df, monthly_df, suffix="_mainline")

                # 打印摘要
                analyzer.print_summary(activity_df, mainline_df)
            else:
                logger.error("原始数据文件不存在")
        else:
            logger.error("活跃度数据文件不存在，请先运行 --analyze")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
