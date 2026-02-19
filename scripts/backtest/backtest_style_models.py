#!/usr/bin/env python3
"""
风格模型回测对比框架

对比9指标版本和观察推导版本的表现
"""

import logging
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


class StyleModelBacktester:
    """风格模型回测对比"""

    def __init__(self):
        self.data_dir = str(get_tushare_root())
        self.output_dir = str(get_data_path("processed", "factors", ensure=True))

        logger.info("风格模型回测对比初始化")

    def load_model_results(self):
        """加载两个模型的结果"""
        logger.info("=" * 70)
        logger.info("加载模型结果...")
        logger.info("=" * 70)

        # 加载9指标版本结果
        file_9indicators = os.path.join(self.output_dir, "style_model_9indicators_results.parquet")
        if os.path.exists(file_9indicators):
            self.results_9indicators = pd.read_parquet(file_9indicators)
            self.results_9indicators["trade_date"] = pd.to_datetime(self.results_9indicators["trade_date"])
            logger.info(f"✓ 9指标版本: {len(self.results_9indicators)} 条记录")
        else:
            logger.error(f"✗ 未找到9指标版本结果: {file_9indicators}")
            return False

        # 加载观察推导版本结果
        file_observation = os.path.join(self.output_dir, "style_model_observation_results.parquet")
        if os.path.exists(file_observation):
            self.results_observation = pd.read_parquet(file_observation)
            self.results_observation["trade_date"] = pd.to_datetime(self.results_observation["trade_date"])
            logger.info(f"✓ 观察推导版本: {len(self.results_observation)} 条记录")
        else:
            logger.error(f"✗ 未找到观察推导版本结果: {file_observation}")
            return False

        return True

    def load_benchmark_data(self):
        """加载基准数据"""
        logger.info("加载基准数据...")

        # 加载指数数据作为基准
        index_file = os.path.join(self.data_dir, "index", "index_ohlc_all.parquet")
        if os.path.exists(index_file):
            self.index_data = pd.read_parquet(index_file)
            self.index_data["trade_date"] = pd.to_datetime(self.index_data["date"])
            logger.info(f"✓ 指数数据: {len(self.index_data)} 条记录")
        else:
            logger.error("✗ 未找到指数数据")
            return False

        return True

    def calculate_returns(self, results_df, benchmark_data):
        """计算收益曲线"""
        # 获取沪深300作为基准
        hs300 = benchmark_data[benchmark_data["code"] == "000300.SH"].copy()
        hs300 = hs300.sort_values("trade_date")

        # 合并数据
        merged = results_df.merge(hs300[["trade_date", "close"]], on="trade_date", how="left")

        # 计算基准收益率
        merged["benchmark_ret"] = merged["close"].pct_change().fillna(0)

        # 计算策略收益率（根据仓位）
        merged["strategy_ret"] = merged["benchmark_ret"] * merged["position"].shift(1).fillna(0)

        # 计算累计收益
        merged["benchmark_cumret"] = (1 + merged["benchmark_ret"]).cumprod() - 1
        merged["strategy_cumret"] = (1 + merged["strategy_ret"]).cumprod() - 1

        return merged

    def calculate_metrics(self, returns):
        """计算绩效指标"""
        metrics = {}

        # 年化收益率
        trading_days = len(returns)
        years = trading_days / 252
        annual_return = (1 + returns["strategy_cumret"].iloc[-1]) ** (1 / years) - 1
        metrics["annual_return"] = annual_return

        # 年化波动率
        annual_vol = returns["strategy_ret"].std() * np.sqrt(252)
        metrics["annual_volatility"] = annual_vol

        # 夏普比率
        risk_free_rate = 0.03  # 假设无风险利率3%
        sharpe_ratio = (annual_return - risk_free_rate) / annual_vol
        metrics["sharpe_ratio"] = sharpe_ratio

        # 最大回撤
        cumret = returns["strategy_cumret"]
        running_max = cumret.expanding().max()
        drawdown = (cumret - running_max) / running_max
        max_drawdown = drawdown.min()
        metrics["max_drawdown"] = max_drawdown

        # 卡尔玛比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        metrics["calmar_ratio"] = calmar_ratio

        # 胜率
        win_rate = (returns["strategy_ret"] > 0).sum() / len(returns["strategy_ret"])
        metrics["win_rate"] = win_rate

        # 平均持仓时间
        avg_position = returns["position"].mean()
        metrics["avg_position"] = avg_position

        # 换手率（简化）
        position_change = returns["position"].diff().abs()
        turnover_rate = position_change.mean()
        metrics["turnover_rate"] = turnover_rate

        return metrics

    def compare_models(self):
        """对比两个模型"""
        logger.info("\n" + "=" * 70)
        logger.info("计算模型表现...")
        logger.info("=" * 70)

        # 计算9指标版本的收益
        merged_9indicators = self.calculate_returns(self.results_9indicators, self.index_data)
        metrics_9indicators = self.calculate_metrics(merged_9indicators)

        # 计算观察推导版本的收益
        merged_observation = self.calculate_returns(self.results_observation, self.index_data)
        metrics_observation = self.calculate_metrics(merged_observation)

        # 对比结果

        # 计算相对表现
        relative_performance = {}
        for key in metrics_9indicators.keys():
            obs_value = metrics_observation[key]
            ind_value = metrics_9indicators[key]
            if ind_value != 0:
                relative_performance[key] = (obs_value - ind_value) / abs(ind_value)
            else:
                relative_performance[key] = 0 if obs_value == 0 else 1

        return {
            "9指标版本": metrics_9indicators,
            "观察推导版本": metrics_observation,
            "相对表现": relative_performance,
            "merged_9indicators": merged_9indicators,
            "merged_observation": merged_observation,
        }

    def print_comparison(self, comparison):
        """打印对比结果"""
        logger.info("\n" + "=" * 70)
        logger.info("风格模型对比结果")
        logger.info("=" * 70)

        print("\n{:<30} {:>15} {:>15} {:>15}".format("指标", "9指标版本", "观察推导版本", "相对表现"))
        print("-" * 90)

        metrics = comparison["9指标版本"].keys()
        for metric in metrics:
            ind_value = comparison["9指标版本"][metric]
            obs_value = comparison["观察推导版本"][metric]
            relative = comparison["相对表现"][metric]

            # 格式化输出
            ind_str = f"{ind_value:.2%}" if "rate" in metric or "ratio" in metric else f"{ind_value:.4f}"
            obs_str = f"{obs_value:.2%}" if "rate" in metric or "ratio" in metric else f"{obs_value:.4f}"
            rel_str = f"{relative:+.2%}"

            print("{:<30} {:>15} {:>15} {:>15}".format(metric, ind_str, obs_str, rel_str))

        # 状态分布对比
        logger.info("\n" + "=" * 70)
        logger.info("状态分布对比")
        logger.info("=" * 70)

        print("\n9指标版本状态分布：")
        state_counts_9 = comparison["merged_9indicators"]["market_state"].value_counts()
        for state, count in state_counts_9.items():
            print(f"  {state}: {count} ({count/len(comparison['merged_9indicators'])*100:.1f}%)")

        print("\n观察推导版本逻辑分布：")
        logic_counts_obs = comparison["merged_observation"]["logic_type"].value_counts()
        for logic, count in logic_counts_obs.items():
            print(f"  {logic}: {count} ({count/len(comparison['merged_observation'])*100:.1f}%)")

    def generate_report(self, comparison):
        """生成对比报告"""
        logger.info("\n" + "=" * 70)
        logger.info("生成对比报告...")
        logger.info("=" * 70)

        report_file = os.path.join(self.output_dir, "style_models_comparison_report.txt")

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("=" * 90 + "\n")
            f.write("风格模型对比报告\n")
            f.write("=" * 90 + "\n\n")

            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("=" * 90 + "\n")
            f.write("绩效指标对比\n")
            f.write("=" * 90 + "\n\n")

            f.write("{:<30} {:>15} {:>15} {:>15}\n".format("指标", "9指标版本", "观察推导版本", "相对表现"))
            f.write("-" * 90 + "\n")

            metrics = comparison["9指标版本"].keys()
            for metric in metrics:
                ind_value = comparison["9指标版本"][metric]
                obs_value = comparison["观察推导版本"][metric]
                relative = comparison["相对表现"][metric]

                ind_str = f"{ind_value:.2%}" if "rate" in metric or "ratio" in metric else f"{ind_value:.4f}"
                obs_str = f"{obs_value:.2%}" if "rate" in metric or "ratio" in metric else f"{obs_value:.4f}"
                rel_str = f"{relative:+.2%}"

                f.write("{:<30} {:>15} {:>15} {:>15}\n".format(metric, ind_str, obs_str, rel_str))

            f.write("\n" + "=" * 90 + "\n")
            f.write("状态分布\n")
            f.write("=" * 90 + "\n\n")

            f.write("9指标版本状态分布：\n")
            state_counts_9 = comparison["merged_9indicators"]["market_state"].value_counts()
            for state, count in state_counts_9.items():
                f.write(f"  {state}: {count} ({count/len(comparison['merged_9indicators'])*100:.1f}%)\n")

            f.write("\n观察推导版本逻辑分布：\n")
            logic_counts_obs = comparison["merged_observation"]["logic_type"].value_counts()
            for logic, count in logic_counts_obs.items():
                f.write(f"  {logic}: {count} ({count/len(comparison['merged_observation'])*100:.1f}%)\n")

            f.write("\n" + "=" * 90 + "\n")
            f.write("结论\n")
            f.write("=" * 90 + "\n\n")

            # 判断哪个版本更好
            annual_return_diff = comparison["观察推导版本"]["annual_return"] - comparison["9指标版本"]["annual_return"]
            sharpe_diff = comparison["观察推导版本"]["sharpe_ratio"] - comparison["9指标版本"]["sharpe_ratio"]
            max_dd_diff = comparison["观察推导版本"]["max_drawdown"] - comparison["9指标版本"]["max_drawdown"]

            if annual_return_diff > 0 and sharpe_diff > 0 and max_dd_diff > 0:
                f.write("结论：观察推导版本在年化收益、夏普比率和最大回撤方面均优于9指标版本。\n")
            elif annual_return_diff > 0:
                f.write("结论：观察推导版本在年化收益方面优于9指标版本。\n")
            elif sharpe_diff > 0:
                f.write("结论：观察推导版本在夏普比率方面优于9指标版本。\n")
            elif max_dd_diff > 0:
                f.write("结论：观察推导版本在最大回撤控制方面优于9指标版本。\n")
            else:
                f.write("结论：9指标版本在某些方面表现更好。\n")

        logger.info(f"✓ 报告已保存: {report_file}")

    def plot_comparison(self, comparison):
        """绘制对比图表"""
        logger.info("\n" + "=" * 70)
        logger.info("绘制对比图表...")
        logger.info("=" * 70)

        try:
            # 设置中文字体
            plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei"]
            plt.rcParams["axes.unicode_minus"] = False

            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle("风格模型对比", fontsize=16, fontweight="bold")

            # 子图1：累计收益曲线
            ax1 = axes[0, 0]
            ax1.plot(
                comparison["merged_9indicators"]["trade_date"],
                comparison["merged_9indicators"]["strategy_cumret"],
                label="9指标版本",
                linewidth=2,
            )
            ax1.plot(
                comparison["merged_observation"]["trade_date"],
                comparison["merged_observation"]["strategy_cumret"],
                label="观察推导版本",
                linewidth=2,
            )
            ax1.set_title("累计收益曲线", fontsize=12, fontweight="bold")
            ax1.set_xlabel("日期")
            ax1.set_ylabel("累计收益")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 子图2：仓位变化
            ax2 = axes[0, 1]
            ax2.plot(
                comparison["merged_9indicators"]["trade_date"],
                comparison["merged_9indicators"]["position"],
                label="9指标版本",
                linewidth=2,
                alpha=0.7,
            )
            ax2.plot(
                comparison["merged_observation"]["trade_date"],
                comparison["merged_observation"]["position"],
                label="观察推导版本",
                linewidth=2,
                alpha=0.7,
            )
            ax2.set_title("仓位变化", fontsize=12, fontweight="bold")
            ax2.set_xlabel("日期")
            ax2.set_ylabel("仓位")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim([0, 1.1])

            # 子图3：回撤曲线
            ax3 = axes[1, 0]
            # 9指标版本回撤
            cumret_9 = comparison["merged_9indicators"]["strategy_cumret"]
            running_max_9 = cumret_9.expanding().max()
            drawdown_9 = (cumret_9 - running_max_9) / running_max_9

            # 观察推导版本回撤
            cumret_obs = comparison["merged_observation"]["strategy_cumret"]
            running_max_obs = cumret_obs.expanding().max()
            drawdown_obs = (cumret_obs - running_max_obs) / running_max_obs

            ax3.plot(comparison["merged_9indicators"]["trade_date"], drawdown_9, label="9指标版本", linewidth=2)
            ax3.plot(comparison["merged_observation"]["trade_date"], drawdown_obs, label="观察推导版本", linewidth=2)
            ax3.set_title("回撤曲线", fontsize=12, fontweight="bold")
            ax3.set_xlabel("日期")
            ax3.set_ylabel("回撤")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.fill_between(comparison["merged_9indicators"]["trade_date"], drawdown_9, 0, alpha=0.3)
            ax3.fill_between(comparison["merged_observation"]["trade_date"], drawdown_obs, 0, alpha=0.3)

            # 子图4：状态/逻辑分布
            ax4 = axes[1, 1]
            states_9 = comparison["merged_9indicators"]["market_state"].value_counts()
            logics_obs = comparison["merged_observation"]["logic_type"].value_counts()

            x = np.arange(len(states_9))
            width = 0.35

            ax4.bar(x - width / 2, states_9.values, width, label="9指标版本")
            ax4.bar(x + width / 2, logics_obs.values, width, label="观察推导版本")

            ax4.set_title("状态/逻辑分布", fontsize=12, fontweight="bold")
            ax4.set_xlabel("状态/逻辑")
            ax4.set_ylabel("次数")
            ax4.set_xticks(x)
            ax4.set_xticklabels(states_9.index, rotation=45, ha="right")
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis="y")

            plt.tight_layout()

            # 保存图表
            plot_file = os.path.join(self.output_dir, "style_models_comparison.png")
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            logger.info(f"✓ 图表已保存: {plot_file}")

            plt.close()

        except Exception as e:
            logger.warning(f"绘制图表时出错: {e}")

    def run(self):
        """执行完整的对比流程"""
        logger.info("\n" + "=" * 70)
        logger.info("开始风格模型对比...")
        logger.info("=" * 70)

        if not self.load_model_results():
            logger.error("加载模型结果失败")
            return None

        if not self.load_benchmark_data():
            logger.error("加载基准数据失败")
            return None

        # 对比模型
        comparison = self.compare_models()

        # 打印对比结果
        self.print_comparison(comparison)

        # 生成报告
        self.generate_report(comparison)

        # 绘制图表
        self.plot_comparison(comparison)

        logger.info("\n" + "=" * 70)
        logger.info("✓ 风格模型对比完成！")
        logger.info("=" * 70)

        return comparison


def main():
    """主函数"""
    backtester = StyleModelBacktester()
    comparison = backtester.run()

    if comparison is not None:
        print("\n" + "=" * 70)
        print("对比完成！")
        print("=" * 70)
        print("\n文件位置：")
        print("- 对比报告: data/processed/factors/style_models_comparison_report.txt")
        print("- 对比图表: data/processed/factors/style_models_comparison.png")


if __name__ == "__main__":
    main()
