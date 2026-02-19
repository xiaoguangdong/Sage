"""
方案4：使用5个周线指标判断沪深300牛熊状态
指标：
1. index_ret_4w: 指数4周收益
2. index_vol_4w: 指数4周波动
3. ma_diff: 20周-60周均线差
4. breadth: 上涨股票占比
5. new_high_ratio: 创新高比例
"""

import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.data._shared.runtime import get_tushare_root

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class WeeklyLabeler:
    """基于周线指标的沪深300打标器"""

    def __init__(self, data_dir=None, output_dir="images/label"):
        """
        初始化打标器

        Args:
            data_dir: 数据目录
            output_dir: 输出目录
        """
        self.data_dir = data_dir or str(get_tushare_root())
        self.output_dir = output_dir

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 加载数据
        self.load_data()

    def load_data(self):
        """加载沪深300指数和成分股数据"""
        print("=" * 80)
        print("加载沪深300指数和成分股数据...")
        print("=" * 80)

        # 1. 读取指数数据
        index_file = os.path.join(self.data_dir, "index", "index_000300_SH_ohlc.parquet")
        self.df_index = pd.read_parquet(index_file)
        self.df_index = self.df_index.sort_values("date").reset_index(drop=True)
        self.df_index["date"] = pd.to_datetime(self.df_index["date"])

        print("✓ 指数数据加载完成")
        print(
            f"  时间范围: {self.df_index['date'].min().strftime('%Y-%m-%d')} 至 {self.df_index['date'].max().strftime('%Y-%m-%d')}"
        )
        print(f"  数据量: {len(self.df_index)} 条记录")

        # 2. 读取成分股数据
        constituents_file = os.path.join(self.data_dir, "constituents", "hs300_constituents_all.parquet")
        self.df_constituents = pd.read_parquet(constituents_file)

        # 获取成分股列表
        self.stock_list = self.df_constituents["ts_code"].unique()
        print("✓ 成分股数据加载完成")
        print(f"  成分股数量: {len(self.stock_list)} 只")

        # 3. 读取成分股日线数据（用于计算breadth和new_high_ratio）
        self.load_stock_data()

    def load_stock_data(self):
        """加载成分股日线数据"""
        print("\n加载成分股日线数据...")

        # 读取daily_basic数据（包含涨跌幅）
        daily_basic_file = os.path.join(self.data_dir, "daily_basic_all.parquet")
        self.df_daily_basic = pd.read_parquet(daily_basic_file)
        self.df_daily_basic["date"] = pd.to_datetime(self.df_daily_basic["date"])

        # 筛选沪深300成分股
        self.df_daily_basic = self.df_daily_basic[self.df_daily_basic["ts_code"].isin(self.stock_list)]

        print("✓ 成分股日线数据加载完成")
        print(f"  数据量: {len(self.df_daily_basic)} 条记录")

    def resample_to_weekly(self):
        """将日线数据重采样为周线数据"""
        print("\n" + "=" * 80)
        print("重采样为周线数据...")
        print("=" * 80)

        # 指数数据重采样
        weekly_index = (
            self.df_index.resample("W-MON", on="date")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "vol": "sum"})
            .dropna()
        )

        weekly_index = weekly_index.reset_index()
        weekly_index.columns = ["date", "open", "high", "low", "close", "vol"]

        print("✓ 指数周线数据生成完成")
        print(
            f"  时间范围: {weekly_index['date'].min().strftime('%Y-%m-%d')} 至 {weekly_index['date'].max().strftime('%Y-%m-%d')}"
        )
        print(f"  数据量: {len(weekly_index)} 周期")

        self.df_weekly = weekly_index

    def calculate_weekly_indicators(self):
        """计算周线指标"""
        print("\n" + "=" * 80)
        print("计算周线指标...")
        print("=" * 80)

        df = self.df_weekly

        # 1. 移动平均（周线）
        df["ma20w"] = df["close"].rolling(window=20).mean()
        df["ma60w"] = df["close"].rolling(window=60).mean()

        # 2. 指数4周收益 (index_ret_4w)
        df["index_ret_4w"] = df["close"].pct_change(4)

        # 3. 指数4周波动 (index_vol_4w)
        df["index_vol_4w"] = df["close"].pct_change().rolling(window=4).std()

        # 4. 20周-60周均线差 (ma_diff)
        df["ma_diff"] = df["ma20w"] - df["ma60w"]
        # 归一化：相对于MA60的比例
        df["ma_diff_norm"] = df["ma_diff"] / df["ma60w"] * 100

        print("✓ 周线指标计算完成")
        print("  MA20w, MA60w, MA_diff")
        print("  index_ret_4w, index_vol_4w")

        self.df_weekly = df

    def calculate_breadth_and_new_high(self):
        """计算市场广度指标（上涨股票占比、创新高比例）"""
        print("\n" + "=" * 80)
        print("计算市场广度指标...")
        print("=" * 80)

        # 按周聚合成分股数据
        weekly_stocks = (
            self.df_daily_basic.resample("W-MON", on="date")
            .agg({"pct_chg": lambda x: np.mean(x), "ts_code": "count"})  # 平均涨跌幅  # 股票数量
            .reset_index()
        )

        weekly_stocks.columns = ["date", "avg_pct_chg", "stock_count"]

        # 计算上涨股票占比
        # 需要原始数据来计算占比
        weekly_rising = (
            self.df_daily_basic.groupby(pd.Grouper(key="date", freq="W-MON"))
            .apply(lambda x: np.sum(x["pct_chg"] > 0) / len(x) if len(x) > 0 else 0)
            .reset_index()
        )

        weekly_rising.columns = ["date", "breadth"]

        # 计算创新高比例（20日新高）
        # 对每只股票计算是否创新高，然后每周聚合
        def calc_new_high_ratio(group):
            # 计算过去20个交易日的高点
            group["high_20d"] = group["high"].rolling(window=20, min_periods=5).max()
            group["is_new_high"] = group["high"] >= group["high_20d"].shift(1)
            return group

        # 由于daily_basic没有high字段，需要从指数数据获取
        # 这里简化处理：使用涨跌幅作为参考
        # 如果涨跌幅 > 5%，认为可能创新高
        weekly_new_high = (
            self.df_daily_basic.groupby(pd.Grouper(key="date", freq="W-MON"))
            .apply(lambda x: np.sum(x["pct_chg"] > 5) / len(x) if len(x) > 0 else 0)
            .reset_index()
        )

        weekly_new_high.columns = ["date", "new_high_ratio"]

        # 合并到周线数据
        self.df_weekly = pd.merge(self.df_weekly, weekly_rising, on="date", how="left")

        self.df_weekly = pd.merge(self.df_weekly, weekly_new_high, on="date", how="left")

        # 填充缺失值
        self.df_weekly["breadth"] = self.df_weekly["breadth"].fillna(0.5)
        self.df_weekly["new_high_ratio"] = self.df_weekly["new_high_ratio"].fillna(0)

        print("✓ 市场广度指标计算完成")
        print("  breadth: 上涨股票占比")
        print("  new_high_ratio: 创新高比例")

    def calculate_weekly_score(self, i):
        """
        计算周线综合评分

        Args:
            i: 当前行索引

        Returns:
            score: 综合评分
        """
        df = self.df_weekly

        # 数据不足
        if i < 60:  # 需要至少60周数据
            return 0

        ret_4w = df["index_ret_4w"].iloc[i]
        vol_4w = df["index_vol_4w"].iloc[i]
        ma_diff = df["ma_diff_norm"].iloc[i]
        breadth = df["breadth"].iloc[i]
        new_high = df["new_high_ratio"].iloc[i]

        # 检查数据有效性
        if pd.isna(ret_4w) or pd.isna(ma_diff) or pd.isna(breadth):
            return 0

        # ========== 综合评分 ==========
        score = 0

        # 1. 指数4周收益 (权重30%)
        # 正收益加分，负收益减分
        if ret_4w > 0.05:  # 4周涨幅>5%
            score += 0.3
        elif ret_4w > 0.02:  # 4周涨幅>2%
            score += 0.2
        elif ret_4w > 0:
            score += 0.1
        elif ret_4w < -0.05:  # 4周跌幅>5%
            score -= 0.3
        elif ret_4w < -0.02:  # 4周跌幅>2%
            score -= 0.2
        else:
            score -= 0.1

        # 2. 指数4周波动 (权重15%，反向)
        # 波动率高减分，波动率低加分
        if vol_4w < 0.02:
            score += 0.15
        elif vol_4w < 0.04:
            score += 0.1
        elif vol_4w > 0.08:
            score -= 0.15
        elif vol_4w > 0.05:
            score -= 0.1

        # 3. MA差值 (权重25%)
        # MA20 > MA60加分，MA20 < MA60减分
        if ma_diff > 5:  # MA20比MA60高5%以上
            score += 0.25
        elif ma_diff > 2:  # MA20比MA60高2%以上
            score += 0.15
        elif ma_diff > 0:
            score += 0.05
        elif ma_diff < -5:  # MA20比MA60低5%以上
            score -= 0.25
        elif ma_diff < -2:  # MA20比MA60低2%以上
            score -= 0.15
        else:
            score -= 0.05

        # 4. 上涨股票占比 (权重20%)
        # 超过60%上涨加分，低于40%下跌减分
        if breadth > 0.7:
            score += 0.2
        elif breadth > 0.6:
            score += 0.15
        elif breadth > 0.5:
            score += 0.1
        elif breadth < 0.3:
            score -= 0.2
        elif breadth < 0.4:
            score -= 0.15
        elif breadth < 0.5:
            score -= 0.1

        # 5. 创新高比例 (权重10%)
        # 创新高股票多加分
        if new_high > 0.3:
            score += 0.1
        elif new_high > 0.2:
            score += 0.07
        elif new_high > 0.1:
            score += 0.05
        else:
            score += 0

        return np.clip(score, -1, 1)

    def label_market_state(self):
        """判断市场状态（牛/熊/震荡）"""
        print("\n" + "=" * 80)
        print("判断市场状态...")
        print("=" * 80)

        df = self.df_weekly
        labels = []
        scores = []

        for i in range(len(df)):
            score = self.calculate_weekly_score(i)
            scores.append(score)

            # 数据不足
            if i < 60:
                labels.append(1)  # 震荡
                continue

            # 获取关键指标
            ret_4w = df["index_ret_4w"].iloc[i]
            ma_diff = df["ma_diff_norm"].iloc[i]
            breadth = df["breadth"].iloc[i]

            if pd.isna(ret_4w) or pd.isna(ma_diff):
                labels.append(1)
                continue

            # ========== 牛市判断 ==========
            bull_conditions = []

            # 必要条件
            if ma_diff > 0:
                bull_conditions.append("MA20 > MA60")
            if ret_4w > 0:
                bull_conditions.append("4周正收益")

            # 加分条件
            if score > 0.6:
                bull_conditions.append("评分>0.6")
            if ret_4w > 0.05:
                bull_conditions.append("4周涨幅>5%")
            if breadth > 0.6:
                bull_conditions.append("上涨占比>60%")

            # 判断牛市（满足必要条件 + 至少2个加分条件）
            if len(bull_conditions) >= 3:
                labels.append(2)  # 牛市
                continue

            # ========== 熊市判断 ==========
            bear_conditions = []

            # 必要条件
            if ma_diff < 0:
                bear_conditions.append("MA20 < MA60")
            if ret_4w < 0:
                bear_conditions.append("4周负收益")

            # 加分条件
            if score < -0.6:
                bear_conditions.append("评分<-0.6")
            if ret_4w < -0.05:
                bear_conditions.append("4周跌幅>5%")
            if breadth < 0.4:
                bear_conditions.append("上涨占比<40%")

            # 判断熊市
            if len(bear_conditions) >= 3:
                labels.append(0)  # 熊市
                continue

            # ========== 震荡整理 ==========
            labels.append(1)  # 震荡

        self.df_weekly["label"] = labels
        self.df_weekly["score"] = scores

        print("✓ 市场状态判断完成")
        print(f"  牛市: {labels.count(2)} 周")
        print(f"  震荡: {labels.count(1)} 周")
        print(f"  熊市: {labels.count(0)} 周")

    def visualize(self):
        """可视化结果"""
        print("\n" + "=" * 80)
        print("生成可视化图表...")
        print("=" * 80)

        df = self.df_weekly

        # 创建图表
        fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

        # 1. 指数价格
        ax1 = axes[0]
        ax1.plot(df["date"], df["close"], label="收盘价", linewidth=1.5, color="black")
        ax1.plot(df["date"], df["ma20w"], label="MA20周", linewidth=1, color="blue", alpha=0.7)
        ax1.plot(df["date"], df["ma60w"], label="MA60周", linewidth=1, color="orange", alpha=0.7)

        # 用颜色标记市场状态
        for i in range(1, len(df)):
            if df["label"].iloc[i] == 2:  # 牛市
                ax1.axvspan(df["date"].iloc[i - 1], df["date"].iloc[i], alpha=0.2, color="green")
            elif df["label"].iloc[i] == 0:  # 熊市
                ax1.axvspan(df["date"].iloc[i - 1], df["date"].iloc[i], alpha=0.2, color="red")

        ax1.set_ylabel("价格", fontsize=12)
        ax1.set_title("沪深300指数 - 周线方案4", fontsize=14, fontweight="bold")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # 2. 市场状态
        ax2 = axes[1]
        colors = ["red" if v == 0 else "white" if v == 1 else "green" for v in df["label"]]
        ax2.scatter(df["date"], df["label"], c=colors, s=50, alpha=0.7, edgecolors="black", linewidth=0.5)
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(["熊市", "震荡", "牛市"])
        ax2.set_ylabel("市场状态", fontsize=12)
        ax2.grid(True, alpha=0.3, axis="y")

        # 3. 综合评分
        ax3 = axes[2]
        ax3.plot(df["date"], df["score"], label="综合评分", linewidth=1.5, color="purple")
        ax3.axhline(y=0.6, color="green", linestyle="--", alpha=0.5, label="牛市阈值")
        ax3.axhline(y=-0.6, color="red", linestyle="--", alpha=0.5, label="熊市阈值")
        ax3.axhline(y=0, color="black", linestyle="-", alpha=0.3, label="中性")
        ax3.set_ylabel("评分", fontsize=12)
        ax3.set_ylim(-1.1, 1.1)
        ax3.legend(loc="upper left")
        ax3.grid(True, alpha=0.3)

        # 4. 关键指标
        ax4 = axes[3]
        ax4_twin = ax4.twinx()

        # 左轴：4周收益
        line1 = ax4.plot(df["date"], df["index_ret_4w"], label="4周收益", linewidth=1.5, color="blue", alpha=0.8)
        ax4.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax4.set_ylabel("4周收益", fontsize=12, color="blue")
        ax4.tick_params(axis="y", labelcolor="blue")

        # 右轴：上涨占比
        line2 = ax4_twin.plot(df["date"], df["breadth"], label="上涨占比", linewidth=1.5, color="orange", alpha=0.8)
        ax4_twin.axhline(y=0.5, color="black", linestyle="--", alpha=0.3)
        ax4_twin.set_ylabel("上涨占比", fontsize=12, color="orange")
        ax4_twin.tick_params(axis="y", labelcolor="orange")
        ax4_twin.set_ylim(0, 1)

        # 合并图例
        lines = line1 + line2
        labels = [ln.get_label() for ln in lines]
        ax4.legend(lines, labels, loc="upper left")

        ax4.grid(True, alpha=0.3)

        # 格式化x轴
        ax4.set_xlabel("日期", fontsize=12)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45)

        plt.tight_layout()

        # 保存图表
        output_file = os.path.join(self.output_dir, "hs300_weekly_labels.png")
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"✓ 图表已保存: {output_file}")

        plt.close()

    def save_results(self):
        """保存结果数据"""
        print("\n" + "=" * 80)
        print("保存结果数据...")
        print("=" * 80)

        output_file = os.path.join(self.output_dir, "hs300_weekly_labels.csv")
        self.df_weekly.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"✓ 数据已保存: {output_file}")
        print(f"  总计: {len(self.df_weekly)} 周期")

    def generate_summary(self):
        """生成统计摘要"""
        print("\n" + "=" * 80)
        print("统计摘要")
        print("=" * 80)

        df = self.df_weekly

        # 按年份统计
        df["year"] = df["date"].dt.year
        yearly_stats = df.groupby("year").agg(
            {
                "label": [
                    lambda x: (x == 2).sum(),  # 牛市
                    lambda x: (x == 1).sum(),  # 震荡
                    lambda x: (x == 0).sum(),  # 熊市
                ],
                "score": ["mean", "std", "min", "max"],
                "index_ret_4w": ["mean", "std"],
                "breadth": ["mean", "std"],
                "ma_diff_norm": ["mean", "std"],
            }
        )

        yearly_stats.columns = [
            "牛市周数",
            "震荡周数",
            "熊市周数",
            "评分均值",
            "评分标准差",
            "评分最小",
            "评分最大",
            "4周收益均值",
            "4周收益标准差",
            "上涨占比均值",
            "上涨占比标准差",
            "MA差均值",
            "MA差标准差",
        ]

        print("\n年度统计:")
        print(yearly_stats)

        # 总体统计
        total_weeks = len(df)
        bull_weeks = (df["label"] == 2).sum()
        bear_weeks = (df["label"] == 0).sum()
        sideways_weeks = (df["label"] == 1).sum()

        print("\n总体统计:")
        print(f"  总周数: {total_weeks}")
        print(f"  牛市: {bull_weeks} 周 ({bull_weeks/total_weeks*100:.1f}%)")
        print(f"  震荡: {sideways_weeks} 周 ({sideways_weeks/total_weeks*100:.1f}%)")
        print(f"  熊市: {bear_weeks} 周 ({bear_weeks/total_weeks*100:.1f}%)")

        # 评分分布
        print("\n评分分布:")
        print(f"  均值: {df['score'].mean():.3f}")
        print(f"  标准差: {df['score'].std():.3f}")
        print(f"  最小值: {df['score'].min():.3f}")
        print(f"  最大值: {df['score'].max():.3f}")

        # 指标统计
        print("\n关键指标统计:")
        print(f"  4周收益: 均值={df['index_ret_4w'].mean():.3f}, 标准差={df['index_ret_4w'].std():.3f}")
        print(f"  4周波动: 均值={df['index_vol_4w'].mean():.3f}, 标准差={df['index_vol_4w'].std():.3f}")
        print(f"  MA差值: 均值={df['ma_diff_norm'].mean():.3f}, 标准差={df['ma_diff_norm'].std():.3f}")
        print(f"  上涨占比: 均值={df['breadth'].mean():.3f}, 标准差={df['breadth'].std():.3f}")
        print(f"  创新高: 均值={df['new_high_ratio'].mean():.3f}, 标准差={df['new_high_ratio'].std():.3f}")

    def run(self):
        """运行完整流程"""
        print("=" * 80)
        print("周线方案4 - 基于周线指标的牛熊判断")
        print("=" * 80)

        # 1. 加载数据
        self.load_data()

        # 2. 重采样为周线
        self.resample_to_weekly()

        # 3. 计算周线指标
        self.calculate_weekly_indicators()

        # 4. 计算市场广度指标
        self.calculate_breadth_and_new_high()

        # 5. 判断市场状态
        self.label_market_state()

        # 6. 可视化
        self.visualize()

        # 7. 保存结果
        self.save_results()

        # 8. 生成摘要
        self.generate_summary()

        print("\n" + "=" * 80)
        print("✓ 完成！")
        print("=" * 80)


if __name__ == "__main__":
    labeler = WeeklyLabeler()
    labeler.run()
