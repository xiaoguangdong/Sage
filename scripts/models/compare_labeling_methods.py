#!/usr/bin/env python3
"""
对比不同的市场状态标注方法
展示不同规则方案的效果
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


class LabelingMethodComparison:
    """标注方法对比"""

    def __init__(self, data_dir=None, output_dir="images/label"):
        self.data_dir = data_dir or str(get_tushare_root())
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.load_data()

    def load_data(self):
        """加载沪深300指数数据"""
        print("=" * 80)
        print("加载沪深300指数数据...")
        print("=" * 80)

        # 读取指数数据
        file_path = os.path.join(self.data_dir, "index", "index_000300_SH_ohlc.parquet")
        self.df = pd.read_parquet(file_path)

        # 按日期升序排序
        self.df = self.df.sort_values("date").reset_index(drop=True)
        self.df["date"] = pd.to_datetime(self.df["date"])

        print("✓ 数据加载完成")
        print(
            f"  时间范围: {self.df['date'].min().strftime('%Y-%m-%d')} 至 {self.df['date'].max().strftime('%Y-%m-%d')}"
        )
        print(f"  数据量: {len(self.df)} 条记录")

    def calculate_indicators(self):
        """计算技术指标"""
        # 日线指标
        self.df["ma20"] = self.df["close"].rolling(window=20).mean()
        self.df["ma60"] = self.df["close"].rolling(window=60).mean()
        self.df["ma250"] = self.df["close"].rolling(window=250).mean()
        self.df["ma120"] = self.df["close"].rolling(window=120).mean()

        # MACD
        self.df["ema12"] = self.df["close"].ewm(span=12, adjust=False).mean()
        self.df["ema26"] = self.df["close"].ewm(span=26, adjust=False).mean()
        self.df["dif"] = self.df["ema12"] - self.df["ema26"]
        self.df["dea"] = self.df["dif"].ewm(span=9, adjust=False).mean()
        self.df["macd"] = 2 * (self.df["dif"] - self.df["dea"])

        # RSI
        delta = self.df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df["rsi"] = 100 - (100 / (1 + rs))

        # ATR
        high_low = self.df["high"] - self.df["low"]
        high_close = np.abs(self.df["high"] - self.df["close"].shift())
        low_close = np.abs(self.df["low"] - self.df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.df["atr"] = tr.rolling(window=14).mean()

        print("\n✓ 技术指标计算完成")

    def resample_weekly(self):
        """按周重采样"""
        df_weekly = (
            self.df.resample("W", on="date")
            .agg({"close": "last", "open": "first", "high": "max", "low": "min", "vol": "sum"})
            .dropna()
        )

        # 计算周线指标
        df_weekly["ma4"] = df_weekly["close"].rolling(window=4).mean()  # ≈MA20日线
        df_weekly["ma12"] = df_weekly["close"].rolling(window=12).mean()  # ≈MA60日线
        df_weekly["ma48"] = df_weekly["close"].rolling(window=48).mean()  # ≈MA240日线

        print(f"✓ 按周重采样: {len(df_weekly)} 周")
        return df_weekly

    def method1_current_rules(self):
        """方案1：当前规则（综合评分，阈值±0.6）"""
        labels = []
        scores = []

        for i in range(len(self.df)):
            if i < 60:
                labels.append(1)
                scores.append(0)
                continue

            # 计算综合评分
            close = self.df["close"].iloc[i]
            ma20 = self.df["ma20"].iloc[i]
            ma60 = self.df["ma60"].iloc[i]
            ma250 = self.df["ma250"].iloc[i]

            if pd.isna(ma250):
                ma250 = ma60

            # 简化版综合评分
            trend_score = 0
            if close > ma20:
                trend_score += 0.4
            if close > ma60:
                trend_score += 0.3
            if close > ma250:
                trend_score += 0.3

            scores.append(trend_score)

            # 原阈值：±0.6
            if trend_score > 0.6:
                labels.append(2)  # 牛市
            elif trend_score < -0.6:
                labels.append(0)  # 熊市
            else:
                labels.append(1)  # 震荡

        return np.array(labels), np.array(scores)

    def method2_lower_threshold(self):
        """方案2：降低阈值（综合评分，阈值±0.2）"""
        labels = []
        scores = []

        for i in range(len(self.df)):
            if i < 60:
                labels.append(1)
                scores.append(0)
                continue

            close = self.df["close"].iloc[i]
            ma20 = self.df["ma20"].iloc[i]
            ma60 = self.df["ma60"].iloc[i]
            ma250 = self.df["ma250"].iloc[i]

            if pd.isna(ma250):
                ma250 = ma60

            # 简化版综合评分
            trend_score = 0
            if close > ma20:
                trend_score += 0.4
            if close > ma60:
                trend_score += 0.3
            if close > ma250:
                trend_score += 0.3

            scores.append(trend_score)

            # 新阈值：±0.2
            if trend_score > 0.2:
                labels.append(2)  # 牛市
            elif trend_score < -0.2:
                labels.append(0)  # 熊市
            else:
                labels.append(1)  # 震荡

        return np.array(labels), np.array(scores)

    def method3_simplified_ma(self):
        """方案3：简化规则（均线排列）"""
        labels = []

        for i in range(len(self.df)):
            if i < 60:
                labels.append(1)
                continue

            ma20 = self.df["ma20"].iloc[i]
            ma60 = self.df["ma60"].iloc[i]
            ma250 = self.df["ma250"].iloc[i]

            if pd.isna(ma250):
                ma250 = ma60

            # 直接使用均线排列
            if ma20 > ma60 > ma250:
                labels.append(2)  # 牛市
            elif ma20 < ma60 < ma250:
                labels.append(0)  # 熊市
            else:
                labels.append(1)  # 震荡

        return np.array(labels), None

    def method4_weekly_simplified(self):
        """方案4：周线数据 + 简化规则"""
        df_weekly = self.resample_weekly()
        labels = []

        for i in range(len(df_weekly)):
            if i < 48:
                labels.append(1)
                continue

            ma4 = df_weekly["ma4"].iloc[i]
            ma12 = df_weekly["ma12"].iloc[i]
            ma48 = df_weekly["ma48"].iloc[i]

            if pd.isna(ma48):
                ma48 = ma12

            # 周线均线排列
            if ma4 > ma12 > ma48:
                labels.append(2)  # 牛市
            elif ma4 < ma12 < ma48:
                labels.append(0)  # 熊市
            else:
                labels.append(1)  # 震荡

        return df_weekly, np.array(labels)

    def compare_methods(self):
        """对比所有方法"""
        print("\n" + "=" * 80)
        print("对比不同标注方法...")
        print("=" * 80)

        self.calculate_indicators()

        # 方法1-3：日线数据
        labels1, scores1 = self.method1_current_rules()
        labels2, scores2 = self.method2_lower_threshold()
        labels3, _ = self.method3_simplified_ma()

        # 方法4：周线数据
        df_weekly, labels4 = self.method4_weekly_simplified()

        # 统计结果
        print("\n" + "=" * 80)
        print("方法对比统计")
        print("=" * 80)

        methods = [
            ("方案1: 当前规则 (阈值±0.6)", labels1),
            ("方案2: 降低阈值 (阈值±0.2)", labels2),
            ("方案3: 简化规则 (均线排列)", labels3),
        ]

        for method_name, labels in methods:
            unique, counts = np.unique(labels, return_counts=True)
            print(f"\n{method_name}:")
            for label, count in zip(unique, counts):
                if label == 0:
                    print(f"  熊市: {count}天 ({count/len(labels)*100:.1f}%)")
                elif label == 1:
                    print(f"  震荡: {count}天 ({count/len(labels)*100:.1f}%)")
                elif label == 2:
                    print(f"  牛市: {count}天 ({count/len(labels)*100:.1f}%)")

        print("\n方案4: 周线简化规则:")
        unique, counts = np.unique(labels4, return_counts=True)
        for label, count in zip(unique, counts):
            if label == 0:
                print(f"  熊市: {count}周 ({count/len(labels4)*100:.1f}%)")
            elif label == 1:
                print(f"  震荡: {count}周 ({count/len(labels4)*100:.1f}%)")
            elif label == 2:
                print(f"  牛市: {count}周 ({count/len(labels4)*100:.1f}%)")

        # 生成对比图表
        self.generate_comparison_chart(labels1, labels2, labels3, scores1, scores2, df_weekly, labels4)

        return methods

    def generate_comparison_chart(self, labels1, labels2, labels3, scores1, scores2, df_weekly, labels4):
        """生成对比图表"""
        print("\n" + "=" * 80)
        print("生成对比图表...")
        print("=" * 80)

        fig, axes = plt.subplots(4, 1, figsize=(16, 12))

        x_values = self.df["date"]

        # ========== 子图1: 方案1（当前规则） ==========
        ax1 = axes[0]
        ax1.plot(x_values, self.df["close"], label="沪深300", linewidth=1.5, color="black", alpha=0.7)
        ax1.plot(x_values, self.df["ma20"], label="MA20", linewidth=1, color="orange", alpha=0.5)
        ax1.plot(x_values, self.df["ma60"], label="MA60", linewidth=1, color="purple", alpha=0.5)
        ax1.plot(x_values, self.df["ma250"], label="MA250", linewidth=1, color="blue", alpha=0.5)

        # 背景色表示状态
        for i in range(1, len(labels1)):
            if labels1[i] == 2:  # 牛市
                ax1.axvspan(x_values[i - 1], x_values[i], color="green", alpha=0.1)
            elif labels1[i] == 0:  # 熊市
                ax1.axvspan(x_values[i - 1], x_values[i], color="red", alpha=0.1)

        ax1.set_ylabel("点位", fontsize=11)
        ax1.set_title("方案1: 当前规则 (阈值±0.6)", fontsize=12, fontweight="bold")
        ax1.legend(loc="upper left", fontsize=9)
        ax1.grid(True, alpha=0.3)

        # ========== 子图2: 方案2（降低阈值） ==========
        ax2 = axes[1]
        ax2.plot(x_values, self.df["close"], label="沪深300", linewidth=1.5, color="black", alpha=0.7)
        ax2.plot(x_values, self.df["ma20"], label="MA20", linewidth=1, color="orange", alpha=0.5)
        ax2.plot(x_values, self.df["ma60"], label="MA60", linewidth=1, color="purple", alpha=0.5)
        ax2.plot(x_values, self.df["ma250"], label="MA250", linewidth=1, color="blue", alpha=0.5)

        for i in range(1, len(labels2)):
            if labels2[i] == 2:  # 牛市
                ax2.axvspan(x_values[i - 1], x_values[i], color="green", alpha=0.1)
            elif labels2[i] == 0:  # 熊市
                ax2.axvspan(x_values[i - 1], x_values[i], color="red", alpha=0.1)

        ax2.set_ylabel("点位", fontsize=11)
        ax2.set_title("方案2: 降低阈值 (阈值±0.2)", fontsize=12, fontweight="bold")
        ax2.legend(loc="upper left", fontsize=9)
        ax2.grid(True, alpha=0.3)

        # ========== 子图3: 方案3（简化规则） ==========
        ax3 = axes[2]
        ax3.plot(x_values, self.df["close"], label="沪深300", linewidth=1.5, color="black", alpha=0.7)
        ax3.plot(x_values, self.df["ma20"], label="MA20", linewidth=1, color="orange", alpha=0.5)
        ax3.plot(x_values, self.df["ma60"], label="MA60", linewidth=1, color="purple", alpha=0.5)
        ax3.plot(x_values, self.df["ma250"], label="MA250", linewidth=1, color="blue", alpha=0.5)

        for i in range(1, len(labels3)):
            if labels3[i] == 2:  # 牛市
                ax3.axvspan(x_values[i - 1], x_values[i], color="green", alpha=0.1)
            elif labels3[i] == 0:  # 熊市
                ax3.axvspan(x_values[i - 1], x_values[i], color="red", alpha=0.1)

        ax3.set_ylabel("点位", fontsize=11)
        ax3.set_title("方案3: 简化规则 (均线排列)", fontsize=12, fontweight="bold")
        ax3.legend(loc="upper left", fontsize=9)
        ax3.grid(True, alpha=0.3)

        # ========== 子图4: 方案4（周线数据） ==========
        ax4 = axes[3]
        x_values_weekly = df_weekly.index
        ax4.plot(x_values_weekly, df_weekly["close"], label="沪深300(周)", linewidth=2, color="black", alpha=0.7)
        ax4.plot(x_values_weekly, df_weekly["ma4"], label="MA4周(≈MA20日)", linewidth=1.5, color="orange", alpha=0.7)
        ax4.plot(x_values_weekly, df_weekly["ma12"], label="MA12周(≈MA60日)", linewidth=1.5, color="purple", alpha=0.7)
        ax4.plot(x_values_weekly, df_weekly["ma48"], label="MA48周(≈MA240日)", linewidth=1.5, color="blue", alpha=0.7)

        for i in range(1, len(labels4)):
            if labels4[i] == 2:  # 牛市
                ax4.axvspan(x_values_weekly[i - 1], x_values_weekly[i], color="green", alpha=0.15)
            elif labels4[i] == 0:  # 熊市
                ax4.axvspan(x_values_weekly[i - 1], x_values_weekly[i], color="red", alpha=0.15)

        ax4.set_ylabel("点位", fontsize=11)
        ax4.set_title("方案4: 周线简化规则", fontsize=12, fontweight="bold")
        ax4.legend(loc="upper left", fontsize=9)
        ax4.grid(True, alpha=0.3)

        # 调整x轴
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        axes[-1].set_xlabel("日期", fontsize=11)

        plt.tight_layout()

        # 保存图表
        output_path = os.path.join(self.output_dir, "labeling_methods_comparison.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✓ 对比图表已保存: {output_path}")

        # 保存对比数据
        comparison_df = self.df.copy()
        comparison_df["label_method1"] = labels1
        comparison_df["label_method2"] = labels2
        comparison_df["label_method3"] = labels3
        comparison_df["score_method1"] = scores1
        comparison_df["score_method2"] = scores2

        output_csv = os.path.join(self.output_dir, "labeling_methods_comparison.csv")
        comparison_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"✓ 对比数据已保存: {output_csv}")


def main():
    """主函数"""
    print("市场状态标注方法对比系统")
    print("=" * 80)

    comparator = LabelingMethodComparison()
    comparator.compare_methods()

    print("\n" + "=" * 80)
    print("对比完成！")
    print("=" * 80)
    print("图表: images/label/labeling_methods_comparison.png")
    print("数据: images/label/labeling_methods_comparison.csv")


if __name__ == "__main__":
    main()
