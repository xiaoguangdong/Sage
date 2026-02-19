"""
测试趋势模型V2 - 2020-2026年沪深300预测
"""

import os
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sage_core.trend.trend_model import TrendModelConfig
from scripts.data._shared.runtime import get_tushare_root

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_hs300_data():
    """加载沪深300指数数据"""
    data_dir = str(get_tushare_root())
    index_file = os.path.join(data_dir, "index", "index_000300_SH_ohlc.parquet")

    df = pd.read_parquet(index_file)
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    # 筛选2020-2026年
    df = df[(df["date"] >= "2020-01-01") & (df["date"] <= "2026-12-31")].reset_index(drop=True)

    print(f"数据范围: {df['date'].min()} 至 {df['date'].max()}")
    print(f"数据量: {len(df)} 条")

    return df


def test_trend_model(df, config_name="default"):
    """测试趋势模型"""

    # 配置字典
    configs = {
        "default": TrendModelConfig(
            confirmation_periods=3,
            exit_tolerance=5,
            min_hold_periods=7,
        ),
        "sensitive": TrendModelConfig(
            confirmation_periods=2,
            exit_tolerance=3,
            min_hold_periods=5,
        ),
        "conservative": TrendModelConfig(
            confirmation_periods=5,
            exit_tolerance=8,
            min_hold_periods=10,
        ),
    }

    config = configs.get(config_name, configs["default"])

    print(f"\n{'='*60}")
    print(f"测试配置: {config_name}")
    print(f"  confirmation_periods: {config.confirmation_periods}")
    print(f"  exit_tolerance: {config.exit_tolerance}")
    print(f"  min_hold_periods: {config.min_hold_periods}")
    print(f"{'='*60}")

    # 创建模型
    from sage_core.trend.trend_model import TrendModelRuleV2

    model = TrendModelRuleV2(config)

    # 预测（获取历史）
    result = model.predict(df, return_history=True)

    # 提取历史状态
    states = result.diagnostics["states"]
    trend_strength = result.diagnostics["trend_strength"]

    # 统计
    state_counts = {0: 0, 1: 0, 2: 0}
    for s in states:
        state_counts[s] += 1

    total = len(states)
    print("\n状态分布:")
    print(f"  RISK_OFF (熊市): {state_counts[0]} 天 ({state_counts[0]/total*100:.1f}%)")
    print(f"  NEUTRAL (震荡): {state_counts[1]} 天 ({state_counts[1]/total*100:.1f}%)")
    print(f"  RISK_ON (牛市): {state_counts[2]} 天 ({state_counts[2]/total*100:.1f}%)")

    # 计算切换次数
    transitions = sum(1 for i in range(1, len(states)) if states[i] != states[i - 1])
    print(f"\n状态切换: {transitions} 次")
    print(f"平均持续: {total/transitions:.1f} 天")

    return states, trend_strength


def visualize_results(df, states_dict):
    """可视化多个配置的结果"""

    n_configs = len(states_dict)
    fig, axes = plt.subplots(n_configs + 1, 1, figsize=(16, 4 * (n_configs + 1)), sharex=True)

    if n_configs == 0:
        axes = [axes]

    # 第一个子图：价格 + 均线
    ax = axes[0]
    ax.plot(df["date"], df["close"], label="沪深300", linewidth=1.5, color="black", alpha=0.8)

    ma20 = df["close"].rolling(20).mean()
    ma60 = df["close"].rolling(60).mean()
    ma120 = df["close"].rolling(120).mean()

    ax.plot(df["date"], ma20, label="MA20", linewidth=1, color="blue", alpha=0.5)
    ax.plot(df["date"], ma60, label="MA60", linewidth=1, color="orange", alpha=0.5)
    ax.plot(df["date"], ma120, label="MA120", linewidth=1, color="green", alpha=0.5)

    ax.set_ylabel("价格", fontsize=11)
    ax.set_title("沪深300指数 + 均线", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # 后续子图：各配置的状态
    for idx, (config_name, (states, trend_strength)) in enumerate(states_dict.items()):
        ax = axes[idx + 1]

        # 绘制价格
        ax.plot(df["date"], df["close"], linewidth=1, color="black", alpha=0.5)

        # 用颜色标记状态
        for i in range(1, len(df)):
            if states[i] == 2:  # RISK_ON
                ax.axvspan(df["date"].iloc[i - 1], df["date"].iloc[i], alpha=0.3, color="green")
            elif states[i] == 0:  # RISK_OFF
                ax.axvspan(df["date"].iloc[i - 1], df["date"].iloc[i], alpha=0.3, color="red")
            # 震荡不着色

        ax.set_ylabel("价格", fontsize=11)
        ax.set_title(f"趋势状态 - {config_name}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # 添加图例
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="green", alpha=0.3, label="RISK_ON (牛市)"),
            Patch(facecolor="white", alpha=0.3, label="NEUTRAL (震荡)"),
            Patch(facecolor="red", alpha=0.3, label="RISK_OFF (熊市)"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

    # 格式化x轴
    axes[-1].set_xlabel("日期", fontsize=12)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

    plt.suptitle("趋势模型V2 - 2020-2026年回测", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # 保存
    output_dir = "images/trend_model"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "trend_model_v2_2020_2026.png")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\n图表已保存: {output_file}")

    plt.show()


def visualize_trend_strength(df, states_dict):
    """可视化趋势强度"""

    n_configs = len(states_dict)
    fig, axes = plt.subplots(n_configs, 1, figsize=(16, 3 * n_configs), sharex=True)

    if n_configs == 1:
        axes = [axes]

    for idx, (config_name, (states, trend_strength)) in enumerate(states_dict.items()):
        ax = axes[idx]

        # 绘制趋势强度
        dates = df["date"]
        ax.plot(dates, trend_strength, linewidth=1.5, color="blue", alpha=0.7, label="趋势强度")

        # 零线（趋势强度仅为辅助参考，状态由硬条件决定）
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

        # 用颜色标记状态
        for i in range(1, len(df)):
            if states[i] == 2:  # RISK_ON
                ax.axvspan(dates.iloc[i - 1], dates.iloc[i], alpha=0.1, color="green")
            elif states[i] == 0:  # RISK_OFF
                ax.axvspan(dates.iloc[i - 1], dates.iloc[i], alpha=0.1, color="red")

        ax.set_ylabel("趋势强度", fontsize=11)
        ax.set_title(f"趋势强度 - {config_name}", fontsize=12, fontweight="bold")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.1, 1.1)

    axes[-1].set_xlabel("日期", fontsize=12)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

    plt.suptitle("趋势强度 - 2020-2026年", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # 保存
    output_dir = "images/trend_model"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "trend_strength_2020_2026.png")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"图表已保存: {output_file}")

    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("趋势模型V2测试 - 2020-2026年沪深300")
    print("=" * 60)

    # 加载数据
    df = load_hs300_data()

    # 测试多个配置
    states_dict = {}

    for config_name in ["default", "sensitive", "conservative"]:
        states, trend_strength = test_trend_model(df, config_name)
        states_dict[config_name] = (states, trend_strength)

    # 可视化
    print("\n生成可视化图表...")
    visualize_results(df, states_dict)
    visualize_trend_strength(df, states_dict)

    print("\n✓ 测试完成！")
